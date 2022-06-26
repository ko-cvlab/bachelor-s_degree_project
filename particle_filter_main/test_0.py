import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
import dlib
import sys
from scipy.spatial import distance
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
from imutils import face_utils

#楕円形分離度フィルタ
def a(img,a,b,c,d):
    axes_in = a,b
    axes_out = c,d
    angle = 0
    esf = EllipticalSeparabilityFilter(axes_in,axes_out,angle)
    epmap,circles = esf.find_circles(img, num_circles=5)   #num_circlesの適切値は？？
    cut_img = esf.cut_img_4(img,circles)     #カットできない画像は黒画像が返される
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return epmap,cut_img,circles

#中心座標を用いて、指定サイズにトリミング
def cut_img(img,center,size):
    x = int(size[0]/2)
    y = int(size[1]/2)
    a= int(center[0])
    b= int(center[1])
    data = img[b-y:b+y,a-x:a+x]
    return data

#開眼度の計算1
def calc_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    eye_ear = (A + B) / (2.0 * C)
    return round(eye_ear, 3)

#開眼度の計算2
def calc_ratio(img,radius):
    im_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    iris_outline = np.argwhere(im_gray==255)
    x = iris_outline[:,1]
    x = np.unique(x)
    length_max = 0
    for i in x:
        try:
            y_index = np.argwhere(iris_outline[:,1] == i)
            y = iris_outline[y_index,0]
            y_mean = np.mean(y)
            y_up = np.max(y[y<y_mean])
            y_down = np.min(y[y>y_mean])
            length = y_down-y_up
            if length_max < length:
                length_max = length
            #print(f"i:{i}の時 , length_max/(radius*2):{length_max/(radius*2)}")
        except ValueError as e:
            #print(f"radius:{round(radius,3)}, {x[0]}から{x[-1]+1}の横軸において:{i}, Error：{e}")
            pass
    return round(length_max/(radius*2),3)

#虹彩の中心座標と虹彩半径の計算
def eye_center(eye):
    A = (eye[1]+eye[4])
    B = (eye[2]+eye[5])
    position = (A + B) / 4
    C = distance.euclidean(eye[1],eye[4])
    D = distance.euclidean(eye[2],eye[5])
    radius = (C + D ) / 4   #目をつぶったとき分離度マップはどうなる？？
    return position,radius

#右目ランドマークのプロット
def plot_landmark(numpy_img,landmark):
    for index,(x,y) in enumerate(landmark,start=36):
        x = landmark[index][0]
        y = landmark[index][1]
        if(index==42):break
        else:cv.circle(numpy_img, (x, y), 5, (255, 255, 0), -1)
    return numpy_img

#トリミングするサイズの計算
def calc_size(eye):
    A = distance.euclidean(eye[0], eye[3])*2
    B = distance.euclidean(eye[1], eye[4])
    C = distance.euclidean(eye[2], eye[5])
    D = (B + C)
    size = (int(A),int(D))
    return size