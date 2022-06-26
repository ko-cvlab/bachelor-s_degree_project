import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from .test_0 import *

class Contour_Iris:
### インスタンス宣言 ###
    def __init__(self,image,radius):
        self.image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        self.im_gray = cv.cvtColor(self.image, cv.COLOR_RGB2GRAY)
        self.radius = radius
        self.point_list = []
        self.contour = clac_contour_iris(self.image,self.radius)
### callインスタンス宣言 ### 
    def __call__(self):
        for i in range(0,360):
            try:
                angle = i
                error = 0
                image,pixel_list,w = self.contour.rotate(angle)
                point_best = self.contour.spline(pixel_list,w)
                self.point_list.append(point_best)
            except IndexError as e:
                #print(f"{i}番目：{e}")
                error = 1
            except UnboundLocalError as e:
                #print(f"{i}番目：{e}")
                error = 1
            except ValueError as e:
                #print(f"{i}番目：{e}")
                error = 1
                
        if len(self.point_list)==0:
            point_best = (self.image.shape[0]/2,self.image.shape[1]/2)
            self.point_list.append(point_best)
            
        point_list = np.array(self.point_list)
        #print(len(point_list))
        x = point_list[:,0]
        y = point_list[:,1]
        y_gaussian = gaussian_filter1d(y, sigma=20,mode='wrap')
        point_list = [np.array(x),np.array(y_gaussian)]
        return point_list