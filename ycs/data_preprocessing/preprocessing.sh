#!/bin/bash
#Path to original dacon data
export DATAPATH="/workspace/volume2/sohyun/open"
#Path to save preprocessed data
export PREPROCESSED="/workspace/volume2/sohyun/preprocessed"

python3 ./draw_label.py
python3 ./rename.py
python3 ./data_crop.py
python3 ./mv_splits.py
