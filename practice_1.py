#%%
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
import dlib
import sys
from scipy.spatial import distance
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw

from pathlib import Path
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import pickle
#%%
Bright_40 = [16,17,18,63,64,65,245,246,247,307,308,309,454,455,552,553,554,555,556] #アノテーションデータ
Bright_60 = [55,56,57,58,119,120,292,293,294,295,343,350,351,578,579]
Dim_40 = [17,43,44,113,114,219,220,272,273,274,371,412,413,533,534,571,572]
Dim_60 = [17,18,54,87,221,296,297,321,322,347,348,431,432,470,471,506,507,559,560]
#%%
len(Bright_60)
#%%
print(y_true[54])
print(y_true[1])
#%%
len(y_true)
#%%
#リスト保存
import pickle
#f = open("/work/Particle_Filter/Experiment/Bright_40cm/Annotations/annotations.txt", 'wb')
#f = open("/work/Particle_Filter/Experiment/Bright_60cm/Annotations/annotations.txt", 'wb')
#f = open("/work/Particle_Filter/Experiment/Dim_40cm/Annotations/annotations.txt", 'wb')
#f = open("/work/Particle_Filter/Experiment/Dim_60cm/Annotations/annotations.txt", 'wb')
list_row = y_true
pickle.dump(list_row, f)
#%%
