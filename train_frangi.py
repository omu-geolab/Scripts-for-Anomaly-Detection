import numpy as np
import os
import argparse
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTModel
from tqdm import tqdm
from PIL import Image
import warnings

# Skimage/MKLの警告を無視
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# 1. データセットとユーティリティ
# ==============================================================================

# 対応する画像拡張子
IMAGE_EXTENSIONS = ['*.jpg', '*.JPG', '*.png', '*.PNG']

class FilteredImageDataset(Dataset):
    """前処理済み画像（正常データ）をロードするためのカスタムデータセット"""
    def __init__(self, image_dir, img_size=(224, 224)):
        self.image_paths = []
        for ext in IMAGE_EXTENSIONS:
            self.image_paths.extend(glob(os.path.join(image_dir, ext)))
        
        # ViTが期待する形式への変換
        self.transform = transforms.Compose([
            transforms.Resize(img_size),  # 入力サイズにリサイズ
            transforms.ToTensor(),        # Tensorに変換 (0-1)
            # ViTの標準的な正規化
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 画像をロードし、RGBに変換
        # 前処理済みの画像はグレースケールだが、ViTの入力に合わせるためRGBとしてロード
        # フィルタリングの結果（1ch）が3chに複製される
        img = Image.open(self.image_paths[idx]).convert("RGB") 
        
        if self.transform:
            img = self.transform(img)
            
        # ダミーラベル（異常検知はラベル不要）
        return img, 0

def load_vit_backbone(device):
    """ViTモデルをロードし、特徴抽出器として設定する。"""
    # ViT-Baseを読み込み、事前学習済み重みをフリーズ
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

# ==============================================================================
# 2. 異常検知モデル学習ロジック
# ==============================================================================

class AnomalyModelTrainer:
    def __init__(self, model_type: str, output_dir: str, batch_size: int, img_size: int):
        self.model_type = model_type
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.img_size = (img_size, img_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_dim = 768 # ViT-Baseの埋め込み次元

    def train_padim(self, dataloader):
        """ViTを特徴抽出器として使用したPaDiMの学習ロジック"""
        vit = load_vit_backbone(self.device)
        
        all_patch_features = []
        
        # 1. パッチ特徴の抽出
        print("[PaDiM] Extracting patch features...")
        for images, _ in tqdm(dataloader, desc="Extracting Features"):
            images = images.to(self.device)
            with torch.no_grad():
                # features: (Batch, Num_Patches+1, Feature_Dim)
                features = vit(images).last_hidden_state 
                # CLSトークン(インデックス0)を除外し、パッチ特徴を抽出
                patch_features = features[:, 1:, :].reshape(-1, self.feature_dim) 
                all_patch_features.append(patch_features.cpu().numpy())
        
        patch_features_np = np.concatenate(all_patch_features, axis=0)
        print(f"[PaDiM] Total patches extracted: {patch_features_np.shape[0]}")

        # 2. 統計情報（平均と共分散）の計算
        print("[PaDiM] Computing mean vector and covariance matrix...")
        
        # 平均ベクトル
        mean_vector = np.mean(patch_features_np, axis=0)
        
        # 共分散行列
        # rowvar=False: 各列が変数を表す (特徴次元が変数となる)
        cov_matrix = np.cov(patch_features_np, rowvar=False) 
        
        # 3. 安定化と逆行列の計算 (PaDiMの核となる部分)
        # 安定化のために微小な値を対角に加算
        cov_matrix_stabilized = cov_matrix + np.eye(self.feature_dim) * 1e-6 
        
        # 逆行列の計算 (擬似逆行列 pinvを使用するとよりロバスト)
        cov_matrix_inv = np.linalg.pinv(cov_matrix_stabilized)
        
        # 4. モデルの保存
        np.save(os.path.join(self.output_dir, "padim_mean_vector.npy"), mean_vector)
        np.save(os.path.join(self.output_dir, "padim_cov_matrix_inv.npy"), cov_matrix_inv)
        
        # ViTモデルの情報（重みは不要だが、モデルのメタデータは必要）を保存
        torch.save({"model_name": "google/vit-base-patch16-224-in21k"}, 
                   os.path.join(self.output_dir, "vit_metadata.pth"))
        
        print(f"✅ PaDiM training complete. Statistics saved to {self.output_dir}")

    def train_vit_center(self, dataloader):
        """ViT Center-Based One-Class Classificationの学習ロジック"""
        vit = load_vit_backbone(self.device)

        all_cls_features = []
        
        # 1. CLSトークン特徴の抽出
        print("[ViT-Center] Extracting CLS features...")
        for images, _ in tqdm(dataloader, desc="Extracting Features"):
            images = images.to(self.device)
            with torch.no_grad():
                # CLSトークン(インデックス0)の特徴を抽出: (Batch, Feature_Dim)
                cls_features = vit(images).last_hidden_state[:, 0, :]
                all_cls_features.append(cls_features.cpu().numpy())
        
        cls_features_np = np.concatenate(all_cls_features, axis=0)
        print(f"[ViT-Center] Total features extracted: {cls_features_np.shape[0]}")

        # 2. 正常クラスの中心（平均）の計算
        print("[ViT-Center] Computing center vector...")
        center_vector = np.mean(cls_features_np, axis=0)
        
        # 3. モデルの保存
        np.save(os.path.join(self.output_dir, "vit_center_vector.npy"), center_vector)
        
        # ViTモデルの情報と重み（特徴抽出器として再利用するため）を保存
        # 特徴抽出器はフリーズされているため、重みも保存
        torch.save(vit.state_dict(), os.path.join(self.output_dir, "vit_backbone_weights.pth"))
        torch.save({"model_name": "google/vit-base-patch16-224-in21k"}, 
                   os.path.join(self.output_dir, "vit_metadata.pth"))

        print(f"✅ ViT Center-Based training complete. Center vector and weights saved to {self.output_dir}")

    def run(self, data_dir):
        """学習を実行するメインメソッド"""
        # データローダーの準備
        dataset = FilteredImageDataset(data_dir, img_size=self.img_size)
        if len(dataset) == 0:
            print(f"エラー: {data_dir} 内に対応する画像ファイルが見つかりませんでした。")
            return
            
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count() // 2 if os.cpu_count() else 4)

        if self.model_type == "PaDiM":
            self.train_padim(dataloader)
        elif self.model_type == "ViT_Base":
            self.train_vit_center(dataloader)
        else:
            print(f"エラー: 無効なモデルタイプ '{self.model_type}' が選択されました。")


# ==============================================================================
# 3. メイン関数と引数パーサー
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="前処理済み画像を用いてPaDiMまたはViT Center-Based OCCモデルを学習するスクリプト。"
    )
    
    # 環境変数の設定 (OMP競合回避のため)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

    parser.add_argument(
        "--train_dir", 
        type=str, 
        required=True, 
        help="前処理済みひび割れ画像（正常データ）を含むディレクトリのパス。"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["PaDiM", "ViT_Base"], 
        required=True, 
        help="使用する異常検知モデルのタイプ ('PaDiM' または 'ViT_Base')。"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./trained_models", 
        help="学習済みモデル（統計情報や重み）を保存するディレクトリのパス。"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32, 
        help="特徴抽出時のバッチサイズ。"
    )
    parser.add_argument(
        "--img_size", 
        type=int, 
        default=224, 
        help="モデルに入力する画像の一辺のサイズ（ViT標準の224を推奨）。"
    )

    args = parser.parse_args()

    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n--- 学習設定 ---")
    print(f"モデル: {args.model} | 入力サイズ: {args.img_size}x{args.img_size} | バッチサイズ: {args.batch_size}")
    
    trainer = AnomalyModelTrainer(
        model_type=args.model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        img_size=args.img_size
    )

    trainer.run(args.train_dir)