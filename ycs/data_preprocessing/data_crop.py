#-*- coding: utf-8 -*-
import os
from PIL import Image
from tqdm import tqdm

def crop_and_save_image_label(image_path, label_path, output_dir, num, x, y, size):
    image = Image.open(image_path)
    label = Image.open(label_path)
    num, tp = num.split(".")

    for i in range(5):
        for j in range(5):
            left = x * i
            upper = y * j
            right = left + size
            lower = upper + size

            cropped_image = image.crop((left, upper, right, lower))
            cropped_label = label.crop((left, upper, right, lower))

            cropped_image.save(f"{output_dir}/crop_image/{num}_{i * 5 + j + 1}.png")
            cropped_label.save(f"{output_dir}/crop_label/{num}_{i * 5 + j + 1}.png")

def main(preprocess_data_path):
    
    input_image_dir = f"{preprocess_data_path}/train_img"
    input_label_dir = f"{preprocess_data_path}/train_label"
    output_dir = preprocess_data_path

    for image_num in tqdm(os.listdir(input_image_dir)):
        image_path = os.path.join(input_image_dir, image_num)
        label_path = os.path.join(input_label_dir, image_num)

        crop_and_save_image_label(image_path, label_path, output_dir, image_num, 200, 200, 224)


if __name__ == "__main__":

  preprocess_data_path = os.environ['PREPROCESSED']
  os.makedirs(os.path.join(preprocess_data_path, 'crop_image'), exist_ok=True)
  os.makedirs(os.path.join(preprocess_data_path, 'crop_label'), exist_ok=True)

  main(preprocess_data_path)