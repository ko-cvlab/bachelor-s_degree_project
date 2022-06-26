__all__ = ['Eye_open_rate']

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

class Eye_open_rate:
    def __init__(self, cut_image,radius,center):
        self.image = cut_image
        self.radius = radius
        self.center = center
        self.h , self.w = self.image.shape[0],self.image.shape[1]
        
    def cut_img(self):
        x,y = self.center
        radius = self.radius
        h,w = self.h , self.w
        #print(f"radius:{radius} , h:{h} , w:{w}")#しっかり受け取れてる、確認済み
        img_1 = self.image[int(x-radius):int(x+radius),int(y-radius):int(y+radius)]
        h_2 , w_2 = img_1.shape[0],img_1.shape[1]
        #print(f" h_2:{h_2} , w_2:{w_2}")
        img_2 = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)
                
        #c = radius/2←これだと、Brightのときしか上手くいかない
        ratio = radius/25
        c = np.mean(img_2)/(2*1/ratio)-1
        bs = int(np.sqrt(img_2.size))
        
        if bs%2==0:
            bs+=1
        img_3 = cv.adaptiveThreshold(img_2, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, blockSize=bs, C=c)
        #ksize=9
        #image = cv.medianBlur(img2,ksize)
        top = int((h-h_2)/2)
        bottom = int((h-h_2)/2)
        left = int((w-w_2)/2)
        right = int((w-w_2)/2)
        output = cv.copyMakeBorder(img_3,top,bottom,left,right,cv.BORDER_CONSTANT,value=255)
        return output
    
    def calc_blackArea(self,image):
        length_list_1 = []
        coord = np.argwhere(image == 0)
        #print(coord)
        if(len(coord) != 0):
            xmin = np.min(coord, axis=0)[1]
            xmax = np.max(coord, axis=0)[1]
            #print("xmin:"+str(xmin)+" , "+"xmax:"+str(xmax))
            if(xmin!=xmax):
                for num in range(xmin,xmax):#最大の縦幅を算出する
                    x = num
                    if (len(np.argwhere(image[:,x]==0))!=0):
                        x_ymin = np.argwhere(image[:,x]==0)[0][0]
                        x_ymax = np.argwhere(image[:,x]==0)[-1][0]
                        #print("xが"+str(x)+"の時:"+str(ymin)+" , "+str(ymax))
                        length = x_ymax-x_ymin
                    else:
                        length=0
                    length_list_1.append(length)
                length_1 = max(length_list_1) 
                rate_1 = round(length_1/(self.radius*2),3)
            else:
                  rate_1 = 0
        else:
              rate_1 = 0
        return rate_1