import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from geomdl.visualization import VisMPL as vis #オブジェクト指向のBスプラインおよびNURBSライブラリである。
from geomdl.fitting import approximate_curve
from test_0 import *
import test_4 as ss

class Eyelid:
    #インスタンス宣言
    def __init__(self,image,position,size,face_parts):
        self.img = image
        self.center = position#パーティクルフィルタの座標
        self.size = size
        self.eye_position = face_parts[36:42]
        self.upper = self.eye_position[0:4]
        self.below = self.eye_position[[0,5,4,3]]
        self.cut = cut_img(self.img,self.center,size)
        
    def phase_0(self):
        upper, below = self.upper, self.below
        center = self.center
        cut = self.cut
        size = self.size
        
        gap_0 = upper - center
        gap_1 = below - center
        gap_0 = np.array(gap_0,dtype=np.int64)
        gap_1 = np.array(gap_1,dtype=np.int64)

        center_2 = np.array(size)/2
        center_2 = np.array(center_2,dtype=np.int64)

        point_2_0 = gap_0 + center_2
        point_2_1 = gap_1 + center_2
        
        cut_3_0 = cut.copy()
        cut_3_1 = cut.copy()

        pts_0 = np.round(point_2_0).astype(np.int32)
        pts_1 = np.round(point_2_1).astype(np.int32)

        cv2.polylines(cut_3_0, [pts_0], False, (255, 255, 255), thickness=1)
        cv2.polylines(cut_3_1, [pts_1], False, (255, 255, 255), thickness=1)
        
        point_3_0 = list(zip(*np.where(cut_3_0==(255,255,255))[0:2]))
        point_3_1 = list(zip(*np.where(cut_3_1==(255,255,255))[0:2]))

        point_4_0 = np.unique(point_3_0,axis=0)
        point_4_1 = np.unique(point_3_1,axis=0)

        for i in range(len(point_4_0)):
            point_4_0[i] = point_4_0[i][::-1]

        for i in range(len(point_4_1)):
            point_4_1[i] = point_4_1[i][::-1]

        point_4_0 = sorted(point_4_0, key=lambda x: x[0])
        point_4_1 = sorted(point_4_1, key=lambda x: x[0])

        x_0 = np.arange(0,len(point_4_0),5)
        x_1 = np.arange(0,len(point_4_1),5)

        point_5_0 = []
        point_5_1 = []

        [point_5_0.append(point_4_0[x_0[i]]) for i in range(len(x_0))]
        [point_5_1.append(point_4_1[x_1[i]]) for i in range(len(x_1))]
        
        point_5_0 = np.append(np.array(point_5_0),point_2_0,axis=0)#顔特徴点を必ず加える
        point_5_1 = np.append(np.array(point_5_1),point_2_1,axis=0)

        point_5_0 = sorted(point_5_0, key=lambda x: x[0])
        point_5_0 =np.unique(point_5_0,axis=0)
        point_5_1 = sorted(point_5_1, key=lambda x: x[0])
        point_5_1 =np.unique(point_5_1,axis=0)
        
        pts_0 = np.round(point_5_0).astype(np.int32)
        pts_1 = np.round(point_5_1).astype(np.int32)
        
        return pts_0,pts_1
        
    def phase_1(self,pts_0,pts_1):
        upper, below = self.upper, self.below
        center = self.center
        cut = self.cut
        degree = 3
        cs = 8
        
        curve_0 = approximate_curve(pts_0.tolist(), degree, ctrlpts_size=cs)
        curve_1 = approximate_curve(pts_1.tolist(), degree, ctrlpts_size=cs)
        
#         cut_5_2 = cut.copy()
        cut_5_3 = cut.copy()

        result_pts = np.array(curve_0.evalpts).astype(np.int32)
        cv2.polylines(cut_5_3, [result_pts], False, (255, 255, 255), thickness=1)

        result_pts = np.array(curve_1.evalpts).astype(np.int32)
        cv2.polylines(cut_5_3, [result_pts], False, (255, 255, 255), thickness=1)
        
        return cut_5_3
    
    def phase_2(self,pts_0,pts_1,cut,draw_img):
        upper, below = self.upper, self.below
        center = self.center
        h,w = cut.shape[0:2]
        center_3 = w/2 , h/2
        index = list(zip(*np.where(cut==(255,255,255))[0:2]))
        index = np.unique(index,axis=0)
        for i in range(len(index)):
            index[i] = index[i][::-1]
        for i in range(len(index)):
            gap_x = index[i][0] - center_3[0]
            gap_y = index[i][1] - center_3[1]
            x = int(center[0] + gap_x)
            y = int(center[1] + gap_y)
            cv2.circle(draw_img, (x, y), 2, (0, 0, 255), 1)
        return draw_img
        
        