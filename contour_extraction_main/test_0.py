import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
import seaborn as sns
from scipy.interpolate import UnivariateSpline
from scipy.signal import argrelmin, argrelmax
from scipy.interpolate import make_interp_spline

class clac_contour_iris:
### インスタンス宣言 ###
    def __init__(self,image,radius):
        self.im_gray_1 = cv.cvtColor(image, cv.COLOR_RGB2GRAY)#取り出す時用
        self.im_gray_2 = self.im_gray_1.copy()#直線を書く用
        self.image = np.where(self.im_gray_2==255,0,self.im_gray_2)#複製したものは白のところを０にする  
        self.center =  np.array([int(image.shape[0]/2),int(image.shape[1]/2)]) 
        self.radius = radius 
        self.Cx = self.center[1]
        self.Cy = self.center[0]
        self.x = self.Cx+radius*3
        self.y = self.Cy
### 回転、前処理関数 ###  
    def rotate(self,angle):
        Cx = self.Cx
        Cy = self.Cy
        x = self.x
        y = self.y
        center = self.center
        image_1 = self.im_gray_1#元のグレー画像,cv.lineすると黒っぽくなるから用意しておく
        image_2 = np.where(image_1==255,0,self.im_gray_2)
        deg = np.deg2rad(angle)
        cos = np.cos(deg)
        sin = np.sin(deg)
        rot_x = int(cos*x-sin*y+(Cx-Cx*cos+Cy*sin))
        rot_y = int(sin*x+cos*y+(Cy-Cx*sin-Cy*cos))
        point = int(rot_y),int(rot_x)
        cv.line(image_2, center[::-1],point[::-1], 255,thickness=1)#該当するところを白く塗りつぶす
        w = np.argwhere(image_2 == 255)
        #180度以上の場合は、ｘ、ｙともに降順になるようにする
        if  angle >= 180:
            w =w[::-1]
        #90-180の場合の時には、あるｘにおいて、ｙは降順にならなければならない！！！！
        if 180 > angle >= 90:
            #print(len(np.unique(w[:,0])))
            arr = np.empty((0,2),int)
            for i in range(len(np.unique(w[:,0]))):
                x = np.unique(w[:,0])[i]
                y = np.argwhere(w[:,0]==x)
                z = np.empty((0,2),int)       
                for j in range(len(y)):  
                    z = np.append(z,w[y[j][0]].reshape((-1,2)),axis=0)
                z = np.sort(z)[::-1]
                arr = np.append(arr,z,axis=0)
            w = arr.copy()
        #270-360の場合の時には、あるｘにおいて、ｙは昇順にならなければならない！！！！
        if 360 > angle >= 270:
            arr = np.empty((0,2),int)
            for i in range(len(np.unique(w[:,0]))):
                x = np.unique(w[:,0])[i]
                y = np.argwhere(w[:,0]==x)
                z = np.empty((0,2),int)       
                for j in range(len(y)):  
                    z = np.append(z,w[y[j][0]].reshape((-1,2)),axis=0)
                z = np.sort(z)
                arr = np.append(arr,z,axis=0)
            w = arr.copy()
            w = w[::-1]
        #ピクセルリストに該当する画素値を入れる
        pixel_list = []
        for t in range(len(w)):
            pixel_list.append(image_1[w[t][0]][w[t][1]])#格納するときは、描画する前のグレー画像
        return image_1,pixel_list,w#image_1,image_2の場合においてのpixel_listを確認してみる
    
### 極値から最適な境界点を見つける関数 ###
    def spline(self,pixel_list,w):
        #スプライン曲線生成
        x = np.arange(len(pixel_list))
        y = np.array(pixel_list)
        model=make_interp_spline(x, y)
        x=np.linspace(1,len(pixel_list),1000)
        y = model(x)
        arg_r_min,arg_r_max=argrelmin(y),argrelmax(y)
        #極小値のリスト作成
        x_2 = x[arg_r_min[0]]
        y_2 = y[arg_r_min[0]]
        min_value = []
        for a,b in zip(x_2,y_2):
            min_value.append([a,b])
        #極大値のリスト作成
        x_3 = x[arg_r_max[0]]
        y_3 = y[arg_r_max[0]]
        max_value = []
        for a,b in zip(x_3,y_3):
            max_value.append([a,b])
        #傾きリスト作成
        inc_list = []
        x_2_list = []
        x_3_list = []
        for i in range(len(min_value)):
            x_4 = min_value[i][0]
            y_4 = min_value[i][1]
            try:
                x_5 = x_3[x_3>x_4][0]
                y_5 = y_3[x_3>x_4][0]
                inc_list.append(y_5-y_4)
                x_2_list.append(x_4)#極小値
                x_3_list.append(x_5)#極大値
            except IndexError as e:#ある極小値において,それより先に極大値がない場合！！！
                inc_list.append(0)
        #傾きリスト、極小値の座標から最適な境界点を見つける
        inc_list = np.array(inc_list)
        index = np.argmax(np.array(inc_list))
        x_2_best = int(x_2_list[index])#x_2は極小値の配列、極小値配列と傾きリストのインデックスは対応している！！！
        x_3_best = int(x_3_list[index])#x_3は極大値の配列
        a_list = (np.sort(inc_list))[::-1]
        flag = True
        item=0
        mean = np.mean(np.array(pixel_list))
        while flag == True:
            if (x_2_best>(self.radius*1.25) or pixel_list[x_2_best]>(mean/2)):#未満があると目をつぶりそうなとき、上瞼の距離が遠いときに不具合が生じる
                item+=1
                number = a_list[item]
                next_index = np.argwhere(inc_list==number)[0]
                x_2_best = int(x_2_list[next_index[0]])
                x_3_best = int(x_3_list[next_index[0]])
            else:
                flag = False
        a = x_2_best
        b = x_3_best+1
        position_best_list = []
        for i in range(int(a),int(b)):
            try:
                gap = (float(pixel_list[i+1])-float(pixel_list[i]))
                position_best_list.append([pixel_list[i],gap])
            except:
                position_best_list.append([0,0])
        position_best_list = np.array(position_best_list)
        value_index = np.argmax(position_best_list[:,1])
        value = position_best_list[value_index]
        index = int(value_index+a) #value[0]は最適な画素値を表す.aを加算するのを忘れないで！！
        point_best = w[index]
        return point_best 