import os
from distutils.dir_util import copy_tree

data_path = os.environ['DATAPATH']
preprocess_data_path = os.environ['PREPROCESSED']
splits_path = os.path.join(preprocess_data_path, 'splits/')

copy_tree(f"./splits", splits_path)