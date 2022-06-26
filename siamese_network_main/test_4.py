__all__ = ['Eye_open_rate']

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

class Eye_open_rate:
    def __init__(self, cut_image,radius,circle,ratio):
        self.image = cut_image
        self.radius = radius
        self.circle = circle
        self.ratio = ratio
                           
    def cut_img(self):
        a,b=self.circle
        radius = self.radius
        data = self.image[int(b-radius):int(b+radius),int(a-radius):int(a+radius)]
        img = cv.cvtColor(data, cv.COLOR_BGR2GRAY)
        #ret,img2=cv.threshold(img, 25, 255, cv.THRESH_BINARY)
        #ret, img2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)#大津の二値化
        c = np.mean(img)/(2*1/self.ratio)-1
        #c = radius/2
        bs = int(np.sqrt(img.size))
        if bs%2==0:
            bs+=1
        img2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, blockSize=bs, C=c)
        #ksize=9
        #image = cv.medianBlur(img2,ksize)
        return img2
                          
    def calk_eye_rate(self,image):
        cx=self.radius
        cy=self.radius
        image_size = self.radius*self.radius*np.pi#円の面積
        x=0
        y=0
        whitePixels=0
        blackPixels=0
        for x in range(2*self.radius):
            for y in range(2*self.radius):
                c = math.sqrt(abs(x-cx)**2+abs(y-cy)**2)
                if c < self.radius or  c == self.radius:#円内だったらカウント判定をする
                    if image[x][y] ==255:
                        whitePixels+=1
                    else:
                        blackPixels+=1
        #blackPixels = image_size - whitePixels
        whiteAreaRatio = (whitePixels/image_size)*100#[%]
        blackAreaRatio = (blackPixels/image_size)*100#[%]
        return whiteAreaRatio,blackAreaRatio
    
    def calc_blackArea(self,bw_image):
        image_size = bw_image.size
        whitePixels = cv.countNonZero(bw_image)
        blackPixels = bw_image.size - whitePixels
        blackAreaRatio = (blackPixels/image_size)*100#[%]
        blackAreaRatio = round(blackAreaRatio,2)
        return blackAreaRatio
    
    def calc_blackArea_2(self,image):
        length_list_1 = []
        length_list_2 = []
        coord = np.argwhere(image == 0)
        #print(coord)
        if(len(coord) != 0):
            ymin = np.min(coord, axis=0)[0]
            ymax = np.max(coord, axis=0)[0]
            xmin = np.min(coord, axis=0)[1]
            xmax = np.max(coord, axis=0)[1]
        
            #print("xmin:"+str(xmin)+" , "+"xmax:"+str(xmax))
            #print("ymin:"+str(ymin)+" , "+"ymax:"+str(ymax))
            if(xmin!=xmax and ymin!=ymax):
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

                for num in range(ymin,ymax):#最大の横幅を算出す
                    y = num
                    #print(np.argwhere(image[y,:]==0))
                    if (len(np.argwhere(image[y,:]==0))!=0):
                        y_xmin = np.argwhere(image[y,:]==0)[0][0]
                        y_xmax = np.argwhere(image[y,:]==0)[-1][0]
                        #print("yが"+str(y)+"の時:"+str(xmin)+" , "+str(xmax))
                        length = y_xmax-y_xmin
                    else:
                        length = 0
                    length_list_2.append(length)

                length_1 = max(length_list_1) 
                lenght_2 = max(length_list_2)
                rate_1 = round((length_1/lenght_2)*100,2)
            else:
                  rate_1 = 0
        else:
              rate_1 = 0
           
        return rate_1
    
    def calc_blackArea_3(self,image):
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
                rate_1 = round((length_1/(self.radius*2))*100,2)
            else:
                  rate_1 = 0
        else:
              rate_1 = 0
           
        return rate_1
    