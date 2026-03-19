import os
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision import transforms, models
import torch
from torch.utils.data import Dataset, DataLoader

class CrackDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')  # グレースケール変換
        image = image.convert("RGB")  # ViT用に3チャンネルに戻す
        if self.transform:
            image = self.transform(image)
        return image

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='グレースケールに変換する正常画像のディレクトリ')
    parser.add_argument('--model_path', type=str, default='model/vit_grayscale.pth', help='保存するモデルのパス')
    parser.add_argument('--center_path', type=str, default='model/center_grayscale.npy', help='特徴ベクトル中心の保存先')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    return parser.parse_args()

def main():
    args = get_args()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    dataset = CrackDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ViTモデル（事前学習済み）読み込み
    model = models.vit_b_16(pretrained=True)
    model.heads = torch.nn.Identity()  # 出力を特徴ベクトルにする
    model.to(device)
    model.eval()

    all_features = []

    with torch.no_grad():
        for epoch in range(args.epochs):
            print(f"Epoch [{epoch+1}/{args.epochs}]")
            for images in tqdm(dataloader):
                images = images.to(device)
                features = model(images)
                all_features.append(features.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    center = np.mean(all_features, axis=0)

    # 保存
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    torch.save(model.state_dict(), args.model_path)
    np.save(args.center_path, center)
    print("✅ モデルと中心特徴ベクトルを保存しました。")

if __name__ == '__main__':
    main()
