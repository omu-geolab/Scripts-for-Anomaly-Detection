import os
import argparse
import re
from PIL import Image

parser = argparse.ArgumentParser(description='Split images based on box coordinates from txt files')
parser.add_argument('-i', '--img_dir', type=str, help='directory containing JPEG images')
parser.add_argument('-t', '--txt_dir', type=str, help='directory containing TXT files with box coordinates')
parser.add_argument('-o', '--output_dir', type=str, help='directory to save output images')
parser.add_argument('-d', '--division_size', type=int, help='division size of original images')
args = parser.parse_args()

def find_box(l_path, width, height, division_size):
    if os.path.exists(l_path):
        f = open(l_path, 'r', encoding='UTF-8')
        datalist = f.readlines()
        box_data = []
        file_name_without_ext = os.path.splitext(l_path)[0]
        file_name = os.path.basename(file_name_without_ext)
        parts = file_name.split('_')
        desired_parts = parts[-2:]
        values = [int(part) for part in desired_parts]

        #分割サイズによって数値の変更が必要　要改善
        for data in datalist:

            '''
            d = [float(x.strip()) for x in data.split(' ')]
            xc = 256 * (float(values[1]) / 256 + (d[1]))
            yc = 256 * (float(values[0]) / 256 + (d[2]))
            wb = 256 * d[3]
            hb = 256 * d[4]
        
            '''
            
            d = [float(x.strip()) for x in data.split(' ')]

            if values[1] < division_size * (float(width)/ division_size - 1):
                xc = division_size * (float(values[1]) / division_size + (d[1]))
                wb = division_size * d[3]
            else:
                xc = division_size * (float(values[1]) / division_size) + (width % division_size) * d[1]
                wb = (width % division_size) * d[3]

            if values[0] < division_size * (float(height)/division_size - 1):
                yc = division_size * (float(values[0]) / division_size + (d[2]))
                hb = division_size * d[4]
            else:
                yc = division_size * (float(values[0]) / division_size) + (height % division_size) * d[2]
                hb = (height % division_size) * d[4]
           
            xl = int(xc - wb / 2) 
            yl = int(yc - hb / 2) 
            xr = int(xc + wb / 2) 
            yr = int(yc + hb / 2) 

            if xr > width:
                xr = width
            
            if yr > height:
                yr = height
            
            box_data.append([xl, yl, xr, yr, d[5], l_path])
        
        return box_data
    else:
        print(l_path + " is not found")
        exit()

def main():
    img_dir = args.img_dir
    txt_dir = args.txt_dir
    output_dir = args.output_dir
    division_size = args.division_size
    mng_filename = "jpg_box.txt"
    mng_filepath = os.path.join(output_dir, mng_filename)

    if not os.path.exists(img_dir):
        print("Directory not found: " + img_dir)
        exit()

    if not os.path.exists(txt_dir):
        print("txtDirectory not found: " + txt_dir)
        exit()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created new directory: {output_dir}")

    txt_list = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]
    image_boxes = {}

    for txt_file in txt_list:
        if txt_file.startswith('.'): # skip hidden files
            continue

        txt_path = os.path.join(txt_dir, txt_file)

        # ファイル名から拡張子を削除
        file_name_without_extension = os.path.splitext(txt_file)[0]

        # ファイル名をアンダースコアで分割
        name_parts = file_name_without_extension.split('_')

        # 元画像名を生成
        if len(name_parts) <= 6:
            image_name = name_parts[0]
        else:
            image_name = '_'.join(name_parts[:-5])

        img_path = os.path.join(img_dir, image_name + '.JPG')
        print(img_path)
        
        if not os.path.exists(img_path):
            print("Image not found for: " + txt_file)
            continue
        
        img = Image.open(img_path)
        width, height = img.size
        
        box_data = find_box(txt_path, width, height, division_size)
        print(box_data)

        if image_name not in image_boxes:
            image_boxes[image_name] = []
        
        image_boxes[image_name].extend(box_data)

    #cropに失敗した回数をカウント
    failure_count = 0
    failed_labels = []

    for image_name, box_data in image_boxes.items():
        img_path = os.path.join(img_dir, image_name + '.JPG')
        img = Image.open(img_path)
        width, height = img.size

        count = 1
        
        for box in box_data:
            xl, yl, xr, yr, _, txt_name = box
            print(xl, yl, xr, yr)

            if xl < xr and yl < yr:
                cropped_img = img.crop((xl, yl, xr, yr))
                output_filename = f'{os.path.splitext(os.path.basename(txt_name))[0]}.jpg'
                output_path = os.path.join(output_dir, output_filename)
                cropped_img.save(output_path)
                print(output_filename)

                #切り出し画像のid 切り出し画像左上x座標 y座標　切り出し画像右下x座標 y座標
                with open(mng_filepath, 'a') as file:
                    file.write('{} {} {} {} {} {}\n'.format(count, box[0], box[1], box[2], box[3], box[4]))
                
                count += 1
            else:
                failure_count += 1
                failed_labels.append((txt_name, xl, yl, xr, yr))
    
    print("trimming complete!")
    print(f"Total crop failures: " + str(failure_count))
    for txt_file, xl, yl, xr, yr in failed_labels:
        print(f"{txt_file}: {xl} {yl} {xr} {yr}")            


if __name__ == "__main__":
    main()
