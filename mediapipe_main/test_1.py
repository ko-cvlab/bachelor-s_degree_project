import numpy as np
import cv2
import math
from numpy import linalg as LA

class Get_Eyelid_Position:
        def  __init__(self, iris_position, dividing_position, p_list, radius):
            self.iris_position = iris_position
            self.dividing_position = dividing_position
            self.p_list = p_list
            self.radius = radius
            
        def getAngle(self, p_0, p_1):
                p_1 = np.array(p_1)
                vec = p_1 - p_0
                rad = np.arctan2(vec[0], vec[1])
                angle = rad*180/np.pi
                return angle

#         def getAngle(self, p_0, p_1):
#                 p_1 = np.array(p_1)
#                 vec = p_1 - p_0
#                 u = np.array([1, 0])
#                 v = vec
#                 i = np.inner(u, v)
#                 n = LA.norm(u) * LA.norm(v)
#                 c = i / n
#                 angle = np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))
#                 if v[1]>0:angle =angle*-1
#                 return angle
            
        def getDistance(self, p_0, p_1):
            u = p_1 - p_0
            distance = np.linalg.norm(u)
            return distance

        def getNearestPosition(self):
            iris_position = self.iris_position
            dividing_position = self.dividing_position
            p_list = self.p_list
            angle_0 = self.getAngle(iris_position,dividing_position)
            angle_list = []
            for i in range(len(self.p_list)):
                angle_1 = self.getAngle(iris_position, self.p_list[i])
                angle_list.append(angle_1)
            idx = np.abs(np.asarray(angle_list) - angle_0).argmin()
            angle_2 = angle_list[idx]
            eyelid_position = p_list[idx]
            return angle_2 , eyelid_position

        def getJudgement(self, eyelid_position):
            radius = self.radius
            iris_position = self.iris_position
            dividing_position = self.dividing_position
            
            dict_0 = self.getDistance(iris_position, eyelid_position)
            dict_1 = self.getDistance(iris_position, dividing_position)
            gap = dict_0 - dict_1
            
            if gap>=-10 and dict_1<=radius*1.2:
                flag = True
            else:
                flag = False
            return flag