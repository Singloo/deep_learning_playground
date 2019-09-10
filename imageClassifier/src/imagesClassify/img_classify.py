'''
File: /Users/origami/Desktop/dl-projects/dl-playground/src/imagesClassify/img_classify.py
Project: /Users/origami/Desktop/dl-projects/dl-playground/src/imagesClassify
Created Date: Monday May 20th 2019
Author: Rick yang tongxue(🍔🍔) (origami@timvel.com)
-----
Last Modified: Wednesday May 22nd 2019 9:18:55 am
Modified By: Rick yang tongxue(🍔🍔) (origami@timvel.com)
-----
'''
from fastai.vision.transform import get_transforms
from numpy import random
from fastai.vision.data import ImageList
import pandas as pd
import os,sys
path = os.path.abspath('../../data/imageClassify')
df = pd.read_csv(path+'/list_attr_celeba_fixed.csv')
tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
src = (ImageList.from_csv(path,'list_attr_celeba.csv',folder='img_align_celeba')
      .split_by_rand_pct(0.2)
       .label_from_df(label_delim=' '))