#-*- coding: utf-8 -*-

import torch, torchvision
import mmseg
import mmcv
import mmengine
import matplotlib.pyplot as plt
import os.path as osp
import os
import numpy as np
from PIL import Image
import cv2
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

from mmengine.runner import Runner
from mmseg.apis import init_model, inference_model, show_result_pyplot

import csv
import pandas as pd
from mmengine import Config
import torch
import argparse

def main():
    cfg = Config.fromfile('./config_ABC/config_C.py')

    # Let's have a look at the final config used for training
    print(f'Config:\n{cfg.pretty_text}')

    # path_mmseg=os.getcwd()
    # path_now=path_mmseg[:-15]

    checkpoint_path = './checkpoints/iter_110670.pth'
    #checkpoint_path = path_mmseg+'/work_dirs/segformer_modelC/iter_110670.pth'
    model = init_model(cfg, checkpoint_path, 'cuda:0')


    test_path= os.path.join(args.data, "test_img")
    os.makedirs(os.path.join(args.data, "test_label"), exist_ok=True)
    output_path = os.path.join(args.data, "test_label")

    image_len = len(os.listdir(test_path))

    #time measure
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # timings=np.zeros((image_len,1))
    timings = []

    time_i = 0

    for image_num in os.listdir(test_path):

        starter.record()

        img = mmcv.imread(test_path+"/"+image_num)
        result = inference_model(model, img)

        ender.record()
        #wait for gpu sync
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        print(f"cur_time of {time_i}th is ", curr_time) #그래서 output을 log.txt에 저장하도록!
        timings.append(curr_time)

        label_tensor=result.pred_sem_seg.data
        label_tensor=label_tensor.cpu()

        np_arr = np.array(label_tensor, dtype=np.uint8)
        np_arr2 = np_arr[0,...]
        img2 = Image.fromarray(np_arr2)
        print(image_num)

        img2.save(os.path.join(output_path, image_num))
        time_i += 1

    print("mean inference time: ", sum(timings)/len(timings))


    def rle_encode(mask):
        mask_gray=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        non_zero_pixels=np.count_nonzero(mask_gray)
        if(non_zero_pixels==0):
            return -1
        else:
            pixels = mask_gray.flatten()
            pixels = np.concatenate([[0], pixels, [0]])
            runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            return ' '.join(str(x) for x in runs)


    image_files=sorted(os.listdir(output_path))

    f=open('../YCS2.csv','w',newline='')

    data=[]
    data.append([])
    data[0].append('img_id')
    data[0].append('mask_rle')

    j=0
    for image_file in image_files:
        j+=1
        data.append([])
        index=str(j-1)
        data[j].append("TEST_"+index.zfill(5))
        image_path = os.path.join(output_path, image_file)
        mask=cv2.imread(image_path)
        name=rle_encode(mask)
        data[j].append(str(name))
        print(j)
        
    
    writer=csv.writer(f)
    writer.writerows(data)
    f.close

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', 
                        type=str, 
                        required=True,
                        help='path to dataset with test image within it')
    
    args = parser.parse_args()
    main()