import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
import dlib
import sys
from scipy.spatial import distance
from PIL import Image, ImageDraw
from imutils import face_utils
from geomdl.fitting import approximate_curve

sys.path.append("/work/Particle_Filter")
# from Test_0 import MultiSeparabilitySnake_improvement as mss
from Test_0 import MultiSeparabilitySnake as mss

class Iris:
    #インスタンス宣言
    def __init__(self, image, radius, p_list):
        self.img = image
        self.radius = radius
        self.p_list = p_list
        
    def phase_0(self):
        eye_img = self.img.copy()
        radius = self.radius
        h,w = eye_img.shape[0:2]
        center = np.array([w/2 , h/2])
        radius_2 = np.array([int(radius), int(radius)])
        theta = math.radians(0)
        pts = mss.make_ellipse_points(center, radius_2, theta)
        #print(f"h,w:{h},{w}, center:{center}, radius:{radius_2}, theta:{theta}")
        degree = 3
        cs = 8
        rect_size = np.array([40, 20])
        curve = approximate_curve(pts.tolist(), degree, ctrlpts_size=cs)
        
        result_curve = mss.separability_snake(eye_img, curve, rect_size, dif_abs=True)
        
        result_pts = np.array(result_curve.evalpts).astype(np.int32)
        iris_img =mss.draw_point_to_img(result_pts, eye_img)
        iris_img = cv.cvtColor(iris_img,cv.COLOR_BGR2RGB)
        return result_pts,iris_img
    
    def phase_1(self,result_pts,draw_img,center_eye,y_before,x_before):
        center_eye_x_1,center_eye_y_1 = center_eye[0],center_eye[1]
        draw_img = mss.draw_point_to_img_2(result_pts, draw_img, center_eye_y_1, center_eye_x_1, y_before, x_before)
        draw_img = cv.cvtColor(draw_img,cv.COLOR_BGR2RGB)
        return draw_img
            
            
            
            
            
            
            
            
            