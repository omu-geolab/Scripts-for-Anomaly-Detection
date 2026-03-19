import numpy as np
import cv2
import os
import argparse
from glob import glob
import torch
from transformers import ViTModel
from scipy.spatial.distance import mahalanobis
import torch.nn.functional as F
from tqdm import tqdm
from matplotlib import pyplot as plt
import warnings

# Skimageの警告を無視
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# 1. ユーティリティ関数（変更なし）
# ==============================================================================

IMAGE_EXTENSIONS = ['*.jpg', '*.JPG', '*.png', '*.PNG']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_DIM = 768

def load_vit_backbone(device):
    """ViTモデルをロードし、特徴抽出器として設定する。"""
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def preprocess_and_load(image_paths: list, img_size: int) -> list:
    """画像をロードし、グレースケール変換とリサイズを行う。"""
    processed_images = []
    target_size = (img_size, img_size)
    
    for path in tqdm(image_paths, desc="Loading and Preprocessing Images"):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        if img is None: continue
        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        processed_images.append(resized_img.astype(np.float32) / 255.0) 
            
    return processed_images

def process_to_vit_input(image_list: list, device: torch.device, img_size: tuple) -> torch.Tensor:
    """0-1正規化されたグレースケール画像をViT入力形式の3チャネルテンソルに変換する。"""
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)
    processed_tensors = []
    for img in image_list:
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        img_tensor = img_tensor.repeat(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        processed_tensors.append(img_tensor.squeeze(0))
    if not processed_tensors:
        return torch.empty(0)
    batch_tensor = torch.stack(processed_tensors).to(device)
    return batch_tensor

# ==============================================================================
# 2. グローバル可視化機能付き PaDiMVisualizer (修正箇所)
# ==============================================================================

class PaDiMVisualizer:
    
    def __init__(self, model_dir: str, img_size: int):
        self.model_dir = model_dir
        self.img_size = (img_size, img_size)
        self.vit = load_vit_backbone(DEVICE)
        self.mean_vector = None
        self.cov_matrix_inv = None
        self.patch_resolution = img_size // 16

    def load_model(self):
        """学習済みPaDiM統計情報をロードする。"""
        mean_path = os.path.join(self.model_dir, "padim_mean_vector.npy")
        cov_inv_path = os.path.join(self.model_dir, "padim_cov_matrix_inv.npy")
        
        if not all(os.path.exists(p) for p in [mean_path, cov_inv_path]):
            raise FileNotFoundError(f"PaDiM model files not found in {self.model_dir}")
            
        self.mean_vector = np.load(mean_path)
        self.cov_matrix_inv = np.load(cov_inv_path)
        print("PaDiM statistics loaded successfully.")

    def calculate_all_anomaly_maps(self, infer_image_paths: list) -> list:
        """全画像に対して推論を実行し、未正規化の異常マップのリストを返す（第1フェーズ）。"""
        
        test_images = preprocess_and_load(infer_image_paths, self.img_size[0])
        image_tensors = process_to_vit_input(test_images, DEVICE, self.img_size)
        
        if image_tensors.numel() == 0:
            return []

        all_anomaly_maps = []
        for i in tqdm(range(image_tensors.shape[0]), desc="Calculating Unnormalized Anomaly Maps"):
            
            # 1. 特徴抽出とマハラノビス距離計算（画像ごとのパッチ）
            with torch.no_grad():
                features = self.vit(image_tensors[i].unsqueeze(0)).last_hidden_state
                test_patch_features = features[0, 1:, :].cpu().numpy()
            
            mahalanobis_distances = []
            for patch_feature in test_patch_features:
                mahalanobis_dist_sq = mahalanobis(patch_feature, self.mean_vector, self.cov_matrix_inv) ** 2
                mahalanobis_distances.append(np.sqrt(mahalanobis_dist_sq))
                
            dist_map = np.array(mahalanobis_distances)
            
            # 2. 異常マップの作成とリサイズ
            map_resolution = self.patch_resolution
            dist_map = dist_map.reshape(map_resolution, map_resolution)
            anomaly_map = cv2.resize(dist_map, self.img_size, interpolation=cv2.INTER_LINEAR)
            
            all_anomaly_maps.append(anomaly_map)
            
        return all_anomaly_maps

    def visualize_and_save(self, original_images: list, all_anomaly_maps: list, global_min: float, global_max: float, output_dir: str):
        """グローバルなmin/max値を使って全画像を正規化・可視化し保存する（第2フェーズ）。"""
        
        for i, anomaly_map in tqdm(enumerate(all_anomaly_maps), total=len(all_anomaly_maps), desc="Generating Global Heatmaps"):
            
            # 1. グローバルな正規化
            # 全画像の最大値と最小値を使って0-1に正規化
            range_val = global_max - global_min
            anomaly_map_norm = (anomaly_map - global_min) / (range_val + 1e-6)
            
            # 2. 可視化結果の生成
            original_path = original_images[i]
            original_img = cv2.imread(original_path)
            if original_img is None: continue
            
            # Matplotlibでヒートマップを作成
            heatmap = plt.cm.jet(anomaly_map_norm)[:, :, :3]
            heatmap = (heatmap * 255).astype(np.uint8)
            
            # 画像のリサイズと重ね合わせ
            original_img_resized = cv2.resize(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), self.img_size)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) # Matplotlibの出力をBGRに戻す
            
            overlay = cv2.addWeighted(original_img_resized, 0.5, heatmap_rgb, 0.5, 0)
            
            # 3. 保存
            filename = os.path.basename(original_path)
            output_path = os.path.join(output_dir, filename)
            # 保存時にRGBをBGRに戻す
            cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
# ==============================================================================
# 3. メイン処理と引数パーサー (修正箇所)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PaDiMモデルの異常度スコア算出根拠を【グローバル正規化】して可視化（ヒートマップ）するスクリプト。"
    )
    
    # ... (引数の設定は変更なし。img_sizeはデフォルト224) ...
    parser.add_argument("--model_dir", type=str, required=True, help="学習済みPaDiMモデルファイル（.npy）が保存されているディレクトリのパス。")
    parser.add_argument("--infer_dir", type=str, required=True, help="推論対象の【前処理（フィルタリング）済み画像】を含むディレクトリのパス。")
    parser.add_argument("--original_dir", type=str, required=True, help="推論対象の画像に対応する【前処理前のオリジナル画像】を含むディレクトリのパス。")
    parser.add_argument("--output_dir", type=str, default="./anomaly_maps", help="生成された異常マップ（ヒートマップ）を保存するディレクトリ。")
    parser.add_argument("--img_size", type=int, default=224, help="モデルに入力する画像の一辺のサイズ（学習時と同じ値を指定）。")


    args = parser.parse_args()

    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. パスの収集
    infer_paths = []
    for ext in IMAGE_EXTENSIONS:
        infer_paths.extend(glob(os.path.join(args.infer_dir, ext)))

    original_paths = [os.path.join(args.original_dir, os.path.basename(p)) for p in infer_paths]
    
    if not infer_paths:
        print(f"エラー: {args.infer_dir} 内に対応する画像ファイルが見つかりませんでした。")
        return

    # 2. Visualizerの初期化とモデルロード
    visualizer = PaDiMVisualizer(args.model_dir, args.img_size)
    try:
        visualizer.load_model()
    except FileNotFoundError as e:
        print(f"致命的なエラー: モデルファイルのロードに失敗しました。{e}")
        return

    # 3. 【第1フェーズ】全画像の異常マップを計算し、グローバルmin/maxを特定
    print("\n--- 第1フェーズ: グローバル統計量の計算 ---")
    all_anomaly_maps = visualizer.calculate_all_anomaly_maps(infer_paths)
    
    if not all_anomaly_maps:
        print("異常マップの生成に失敗しました。")
        return

    # 全マップから最小値と最大値を抽出（これがグローバルな正規化基準となる）
    global_min = np.min([m.min() for m in all_anomaly_maps])
    global_max = np.max([m.max() for m in all_anomaly_maps])
    
    print(f"グローバル最小スコア (Global Min): {global_min:.4f}")
    print(f"グローバル最大スコア (Global Max): {global_max:.4f}")

    # 4. 【第2フェーズ】グローバルなmin/maxを使って可視化
    print("\n--- 第2フェーズ: グローバル正規化による可視化 ---")
    visualizer.visualize_and_save(original_paths, all_anomaly_maps, global_min, global_max, args.output_dir)
    print("\n可視化処理が完了しました。")


if __name__ == "__main__":
    main()