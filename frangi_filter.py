import os
import argparse
import cv2
import numpy as np
from skimage.filters import frangi
from skimage import img_as_ubyte
from tqdm import tqdm

def apply_frangi(input_path, output_path):
    # 画像をグレースケールで読み込み
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {input_path}")

    # Frangi フィルタを適用
    frangi_img = frangi(img)

    # 0-1 の範囲なので 8bit に変換
    frangi_img = img_as_ubyte(frangi_img)

    # 出力ディレクトリが存在しなければ作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存
    cv2.imwrite(output_path, frangi_img)

def main():
    parser = argparse.ArgumentParser(description="Apply Frangi filter to all images in a directory")
    parser.add_argument("-i", "--input_dir", required=True, help="Input directory containing images")
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory for processed images")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    # サポートする拡張子
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    # 入力ディレクトリ内のファイルを処理
    for root, _, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Processing {root}"):
            if any(file.lower().endswith(ext) for ext in exts):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                try:
                    apply_frangi(input_path, output_path)
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    main()
