__all__ = ['Discrimination']

__all__ = ['Divide']

from .test_2 import *
import torch
import torchvision
import torchvision.transforms as transforms
import cv2 as cv
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import glob
from collections import defaultdict
import random

class Divide:
    def __init__(self, cut_image):
        self.image = cut_image
        self.cut_num = len(cut_image)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.rootdir = "/work/Particle_Filter/Siamese_Network"
        self.yes_img_path = glob.glob(self.rootdir+"/image_9/image_9_2(MTCNNによるy_n画像)/y/*.png")
        self.yes_num=len(self.yes_img_path)
        
    def discriminate(self):
        model = Siamese()
        net_path = '/work/Particle_Filter/particle_fileter_main/model_5.pth'#パラメータアップロード
        model.load_state_dict(torch.load(net_path, map_location=torch.device('cpu')))
        
        t=0
        label=[]
        #output_list = defaultdict(list)
        for t in range(self.cut_num):
            #print(self.image[t])
            mean = np.mean(self.image[t], axis=(1, 2))
            if(mean.any() == True):
                yes_count=0#画像毎にyes_countをリセットする！
                #print(self.image[t].shape)
                image = cv.cvtColor(self.image[t], cv.COLOR_BGR2RGB) #ここでカラー変換しないと、BGRのまま推測してしまう
                image1 = self.transform(image)
                w=0

                for w in range(10):#適当なyes画像10に対しての差異をとる！！！
                    x_random = random.randrange(self.yes_num)
                    image2_1 = cv.imread(self.yes_img_path[x_random])
                    image2 = self.transform(image2_1)
                    output = model.forward(image1[None,...], image2[None,...])
                    output = abs(output.item())
                    #output_list[t].append(output) 
                    if(output < 3):              #適切な閾値は？？？
                        yes_count=yes_count+1
                    else:
                        continue
                if(yes_count >= (10/2)):
                    label.append(1)
                else:
                    label.append(0)
                    
            else:
                label.append(0)
                
        return label