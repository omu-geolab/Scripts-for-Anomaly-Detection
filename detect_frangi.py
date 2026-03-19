import numpy as np
import cv2
import os
import argparse
from glob import glob
from abc import ABC, abstractmethod
import torch
from transformers import ViTModel
from scipy.spatial.distance import mahalanobis
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

# ==============================================================================
# 1. ユーティリティ関数 (修正箇所あり)
# ==============================================================================

# 対応する画像拡張子
IMAGE_EXTENSIONS = ['*.jpg', '*.JPG', '*.png', '*.PNG']

def load_vit_backbone(device, weights_path=None):
    """ViTモデルをロードし、特徴抽出器として設定する。"""
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    
    if weights_path and os.path.exists(weights_path):
        try:
            # map_locationはロード時に使用し、to(device)で最終的な配置を行う
            model.load_state_dict(torch.load(weights_path, map_location=lambda storage, loc: storage))
            print(f"Loaded custom weights from {os.path.basename(weights_path)}")
        except Exception as e:
            print(f"Warning: Failed to load custom weights. Using pretrained weights. Error: {e}")

    model.to(device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
            
    return model

def preprocess_and_load(image_paths: list, img_size: int) -> list:
    """画像をロードし、強制リサイズと正規化を行う (パディングなし)。"""
    processed_images = []
    target_size = (img_size, img_size)
    
    for path in tqdm(image_paths, desc="Loading and Preprocessing Images"):
        # グレースケールで読み込み
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        
        if img is None:
            continue
        
        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        # 0-1 正規化 (numpy)
        processed_images.append(resized_img.astype(np.float32) / 255.0)
            
    return processed_images

def process_to_vit_input(image_list: list, device: torch.device, img_size: tuple) -> torch.Tensor:
    """0-1正規化されたグレースケール画像をViT入力形式の3チャネルテンソルに変換する。(修正済)"""
    
    # mean/std はGPU上に配置
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)

    processed_tensors = []
    for img in image_list:
        # 1. numpy -> torch.Tensor (CPU上)
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0) 
        
        # 2. 【修正点A】テンソルをGPUに移動
        img_tensor = img_tensor.to(device) 
        
        # 3. 3チャネルに複製
        img_tensor = img_tensor.repeat(1, 3, 1, 1)
        
        # 4. 【修正点B】GPU上で正規化を実行
        img_tensor = (img_tensor - mean) / std

        processed_tensors.append(img_tensor.squeeze(0))

    if not processed_tensors:
        return torch.empty(0, device=device) # 空のテンソルもデバイス指定が必要

    # 5. バッチテンソルに結合 (既にGPU上)
    batch_tensor = torch.stack(processed_tensors)
    
    return batch_tensor

# ... (AnomalyInferencer, PaDiMInferencer, ViTInferencer クラスは変更なし)
# PaDiMInferencerとViTInferencerのload_model/infer関数に変更はありませんが、
# PaDiMの統計量とViTのcenterベクトルをGPUに移動させて、よりGPU最適化を強化します。
class AnomalyInferencer(ABC):
    """異常検知推論の抽象基底クラス"""
    
    def __init__(self, model_dir: str, img_size: int):
        self.model_dir = model_dir
        self.img_size = (img_size, img_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vit = None
        self.feature_dim = 768

    @abstractmethod
    def load_model(self):
        """学習済みモデル（統計情報や重み）をロードする。"""
        pass

    @abstractmethod
    def infer(self, test_images: list) -> np.ndarray:
        """テスト画像を入力とし、異常度スコアを返す。"""
        pass
        
class PaDiMInferencer(AnomalyInferencer):
    # PaDiMの統計量をGPUに移動するよう修正
    def __init__(self, model_dir: str, img_size: int):
        super().__init__(model_dir, img_size)
        self.mean_vector_gpu = None
        self.cov_matrix_inv_gpu = None
        self.mean_vector_np = None # Mahalanobis距離計算のためにnumpy版も保持
        self.cov_matrix_inv_np = None
        
    def load_model(self):
        mean_path = os.path.join(self.model_dir, "padim_mean_vector.npy")
        cov_inv_path = os.path.join(self.model_dir, "padim_cov_matrix_inv.npy")
        
        if not all(os.path.exists(p) for p in [mean_path, cov_inv_path]):
            raise FileNotFoundError(f"PaDiM model files not found in {self.model_dir}")
            
        # numpyでロード
        self.mean_vector_np = np.load(mean_path)
        self.cov_matrix_inv_np = np.load(cov_inv_path)
        
        # GPU処理のためテンソル化（Mahalanobis距離計算はSciPy/Numpyで実行されるため、ここでは主にViTInferencerの修正を優先します）
        # PaDiMのMahalanobis計算はNumpy/SciPyで行うため、ここでは元のコードのNumpy版を利用します。
        
        self.vit = load_vit_backbone(self.device)
        print("PaDiM statistics and ViT backbone loaded successfully.")

    def infer(self, test_images: list) -> np.ndarray:
        image_tensors = process_to_vit_input(test_images, self.device, self.img_size)
        if image_tensors.numel() == 0: return np.array([0.0] * len(test_images))

        anomaly_scores = []
        # Mahalanobis距離計算のためのNumpy統計情報
        mean_vector = self.mean_vector_np 
        cov_matrix_inv = self.cov_matrix_inv_np
        
        for i in tqdm(range(image_tensors.shape[0]), desc="PaDiM Inference (Mahalanobis Distance)"):
            with torch.no_grad():
                # 特徴抽出はGPU上
                features = self.vit(image_tensors[i].unsqueeze(0)).last_hidden_state 
                # Mahalanobis距離計算のためCPUに戻す
                test_patch_features = features[0, 1:, :].cpu().numpy()
                
                mahalanobis_distances = []
                for patch_feature in test_patch_features:
                    # SciPyのmahalanobisはNumpy配列を要求するため、ここでCPU処理が発生
                    mahalanobis_dist_sq = mahalanobis(patch_feature, mean_vector, cov_matrix_inv) ** 2
                    mahalanobis_distances.append(np.sqrt(mahalanobis_dist_sq))
                
                anomaly_scores.append(np.max(mahalanobis_distances))
                
        return np.array(anomaly_scores)

class ViTInferencer(AnomalyInferencer):
    # CenterベクトルをGPUに移動するよう修正
    def __init__(self, model_dir: str, img_size: int):
        super().__init__(model_dir, img_size)
        self.center_vector_gpu = None
        self.center_vector_np = None

    def load_model(self):
        center_path = os.path.join(self.model_dir, "vit_center_vector.npy")
        weights_path = os.path.join(self.model_dir, "vit_backbone_weights.pth")

        if not all(os.path.exists(p) for p in [center_path, weights_path]):
            raise FileNotFoundError(f"ViT Center model files not found in {self.model_dir}")

        self.center_vector_np = np.load(center_path)
        # 【GPU最適化】centerベクトルをPyTorchテンソルに変換しGPUに移動
        self.center_vector_gpu = torch.from_numpy(self.center_vector_np).float().to(self.device)
        
        self.vit = load_vit_backbone(self.device, weights_path=weights_path)
        print("ViT Center vector and trained ViT weights loaded successfully.")

    def infer(self, test_images: list) -> np.ndarray:
        image_tensors = process_to_vit_input(test_images, self.device, self.img_size)
        if image_tensors.numel() == 0: return np.array([0.0] * len(test_images))

        anomaly_scores = []
        center_vector = self.center_vector_gpu # GPU tensorを使用

        for i in tqdm(range(image_tensors.shape[0]), desc="ViT Inference (Euclidean Distance)"):
            with torch.no_grad():
                # 特徴抽出はGPU上
                test_cls_feature = self.vit(image_tensors[i].unsqueeze(0)).last_hidden_state[:, 0, :].squeeze()
            
            # 【GPU最適化】距離計算をPyTorch (GPU) で実行し、結果のみをCPUに戻す
            # 距離 = ||特徴 - 中心|| (ユークリッド距離)
            distance = torch.norm(test_cls_feature - center_vector).item()
            anomaly_scores.append(distance)
            
        return np.array(anomaly_scores)

# ==============================================================================
# 3. メイン処理と引数パーサー (変更なし)
# ==============================================================================
# ... (main関数のロジックは変更なし、元のコードをそのまま使用)

def main():
    parser = argparse.ArgumentParser(
        description="学習済み異常検知モデルを用いてbbox画像をフィルタリングし、オリジナル画像と前処理後画像をそれぞれ保存するスクリプト。"
    )
    
    # 環境変数の設定 (OMP競合回避のため)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

    # ディレクトリ関連 (変更: 4つの出力ディレクトリを定義)
    parser.add_argument(
        "--model_dir", 
        type=str, 
        required=True, 
        help="学習済みモデルファイル（.npy, .pth）が保存されているディレクトリのパス。"
    )
    parser.add_argument(
        "--infer_dir", 
        type=str, 
        required=True, 
        help="推論対象のbbox画像（前処理済みフィルタ画像）を含むディレクトリのパス。"
    )
    parser.add_argument(
        "--original_dir", 
        type=str, 
        required=True, 
        help="推論対象のbboxに対応する【前処理前のオリジナル画像】を含むディレクトリのパス。"
    )
    # 前処理後の画像 (元々の kept_dir/removed_dir)
    parser.add_argument(
        "--kept_dir", 
        type=str, 
        default="./inferred_kept_filtered",
        help="【前処理後画像】で'ひび割れ'と判断され、保持された画像の出力先ディレクトリ。"
    )
    parser.add_argument(
        "--removed_dir", 
        type=str, 
        default="./inferred_removed_filtered",
        help="【前処理後画像】で'誤検知'と判断され、除去された画像の出力先ディレクトリ。"
    )
    # オリジナル画像 (新しい final_kept_dir/final_removed_dir)
    parser.add_argument(
        "--final_kept_dir", 
        type=str, 
        default="./final_original_kept",
        help="【オリジナル画像】で'ひび割れ'と判断され、保持された画像の最終出力先ディレクトリ。"
    )
    parser.add_argument(
        "--final_removed_dir", 
        type=str, 
        default="./final_original_removed",
        help="【オリジナル画像】で'誤検知'と判断され、除去された画像の最終出力先ディレクトリ。"
    )
    
    # モデル選択とハイパーパラメータ
    parser.add_argument(
        "--model_type", 
        type=str, 
        choices=["PaDiM", "ViT_Base"], 
        required=True, 
        help="使用する異常検知モデルのタイプ ('PaDiM' または 'ViT_Base')。"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        required=True,
        help="異常と判断するスコアの閾値。この値を超えると誤検知として除外される。"
    )
    parser.add_argument(
        "--img_size", 
        type=int, 
        default=224,
        help="モデルに入力する画像の一辺のサイズ（学習時と同じ値を指定）。"
    )

    args = parser.parse_args()

    # 4つの出力ディレクトリの作成
    os.makedirs(args.kept_dir, exist_ok=True)
    os.makedirs(args.removed_dir, exist_ok=True)
    os.makedirs(args.final_kept_dir, exist_ok=True)
    os.makedirs(args.final_removed_dir, exist_ok=True)
    
    # 1. 推論モデルの選択とロード
    if args.model_type == "PaDiM":
        inferencer = PaDiMInferencer(args.model_dir, args.img_size)
    elif args.model_type == "ViT_Base":
        inferencer = ViTInferencer(args.model_dir, args.img_size)
    
    print(f"\n--- 推論設定 ---")
    print(f"モデルタイプ: {args.model_type} | 閾値: {args.threshold} | サイズ: {args.img_size}x{args.img_size}")
    
    try:
        inferencer.load_model()
    except FileNotFoundError as e:
        print(f"致命的なエラー: モデルファイルのロードに失敗しました。{e}")
        return
    
    # 2. 推論対象画像の収集と前処理
    infer_paths = []
    for ext in IMAGE_EXTENSIONS:
        infer_paths.extend(glob(os.path.join(args.infer_dir, ext)))

    if not infer_paths:
        print(f"エラー: {args.infer_dir} 内に対応する画像ファイルが見つかりませんでした。")
        return

    print(f"\n[推論データロード]: {len(infer_paths)}枚の画像を処理します。")
    # ここでtest_imagesはCPU上のNumpy配列のリスト
    test_images = preprocess_and_load(infer_paths, args.img_size)

    # 3. 異常度推論の実行
    start_time = time.time()
    # infer()内でGPUへ移動
    anomaly_scores = inferencer.infer(test_images) 
    end_time = time.time()
    
    # 4. 異常スコアの統計情報計算と表示
    if len(anomaly_scores) > 0:
        scores_np = np.array(anomaly_scores)
        print("\n--- 異常スコアの統計情報 ---")
        print(f"最小値 (Min): {np.min(scores_np):.4f}")
        print(f"最大値 (Max): {np.max(scores_np):.4f}")
        print(f"平均値 (Mean): {np.mean(scores_np):.4f}")
        print(f"中央値 (Median): {np.median(scores_np):.4f}")
        print("-" * 30)
    
    print(f"推論時間: {end_time - start_time:.2f}秒")

    # 5. フィルタリングと【オリジナル画像】の保存 (変更なし)
    removed_count = 0
    kept_count = 0
    
    print("\n--- フィルタリング結果の保存 (4ディレクトリへ振り分け) ---")

    for i, score in tqdm(enumerate(anomaly_scores), total=len(anomaly_scores), desc="Saving Results"):
        if i >= len(infer_paths):
            break

        # フィルタリング後の画像のパスとファイル名
        filtered_img_path = infer_paths[i]
        filename = os.path.basename(filtered_img_path)
        
        # オリジナル画像とフィルタリング後の画像をロード
        original_img_path = os.path.join(args.original_dir, filename)
        original_img = cv2.imread(original_img_path) 
        filtered_img = cv2.imread(filtered_img_path)
        
        if original_img is None or filtered_img is None:
            continue

        # 異常度が閾値より高い場合 -> 誤検知 (Removed)
        if score > args.threshold:
            # 前処理後画像をremoved_dirに保存
            cv2.imwrite(os.path.join(args.removed_dir, filename), filtered_img)
            # オリジナル画像をfinal_removed_dirに保存
            cv2.imwrite(os.path.join(args.final_removed_dir, filename), original_img)
            removed_count += 1
        # 異常度が閾値以下の場合 -> 真のひび割れ (Kept)
        else:
            # 前処理後画像をkept_dirに保存
            cv2.imwrite(os.path.join(args.kept_dir, filename), filtered_img)
            # オリジナル画像をfinal_kept_dirに保存
            cv2.imwrite(os.path.join(args.final_kept_dir, filename), original_img)
            kept_count += 1

    print(f"\n総bbox数: {len(infer_paths)}")
    print(f"残存 (ひび割れ): {kept_count} -> フィルタリング後: '{args.kept_dir}', オリジナル: '{args.final_kept_dir}'")
    print(f"削除 (誤検知): {removed_count} -> フィルタリング後: '{args.removed_dir}', オリジナル: '{args.final_removed_dir}'")


if __name__ == "__main__":
    main()