import os
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms, models
import torch
from tqdm import tqdm
import shutil

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='入力画像ディレクトリ')
    parser.add_argument('--model_path', type=str, required=True, help='学習済みViTモデルのパス')
    parser.add_argument('--center_path', type=str, required=True, help='中心特徴ベクトルのパス')
    parser.add_argument('--out_normal_dir', type=str, required=True, help='正常（ひび割れ）画像の保存先')
    parser.add_argument('--out_abnormal_dir', type=str, required=True, help='異常（ゴミなど）画像の保存先')
    parser.add_argument('--threshold', type=float, required=True, help='異常度のしきい値')
    return parser.parse_args()

def main():
    args = get_args()

    os.makedirs(args.out_normal_dir, exist_ok=True)
    os.makedirs(args.out_abnormal_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    model = models.vit_b_16(pretrained=True)
    model.heads = torch.nn.Identity()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    center = np.load(args.center_path)

    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_name in tqdm(image_files):
        image_path = os.path.join(args.input_dir, image_name)
        image = Image.open(image_path).convert('L')  # グレースケール変換
        image = image.convert("RGB")  # ViTのために3チャンネルに複製
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = model(image_tensor).cpu().numpy().squeeze()
        score = np.linalg.norm(features - center)

        if score < args.threshold:
            shutil.copy(image_path, os.path.join(args.out_normal_dir, image_name))
        else:
            shutil.copy(image_path, os.path.join(args.out_abnormal_dir, image_name))

    print("✅ 推論完了：正常画像 → {}, 異常画像 → {}".format(
        args.out_normal_dir, args.out_abnormal_dir
    ))

if __name__ == '__main__':
    main()
