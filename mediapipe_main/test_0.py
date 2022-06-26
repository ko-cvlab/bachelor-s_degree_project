import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import sys
from mediapipe.framework.formats import landmark_pb2
from scipy.spatial import distance
sys.path.append("/work/Particle_Filter")

#mediapipeの初期設定
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils 
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def calc_position_eyelid_right(face_landmarks,draw_img):
    img_height,img_width,_ = draw_img.shape
    eyelid_position = []
    landmark_index = [33,246,161,159,158,157,173,133,155,154,153,145,144,163,7]
    for i in range(len(landmark_index)):
        x = face_landmarks.landmark[landmark_index[i]].x * img_width
        y = face_landmarks.landmark[landmark_index[i]].y * img_height
        position = x,y
        eyelid_position.append(position)
    return eyelid_position

def calc_position_eyelid_right(face_landmarks,draw_img):
    img_height,img_width,_ = draw_img.shape
    eyelid_position = []
    landmark_index = [33,246,161,159,158,157,173,133,155,154,153,145,144,163,7]
    for i in range(len(landmark_index)):
        x = face_landmarks.landmark[landmark_index[i]].x * img_width
        y = face_landmarks.landmark[landmark_index[i]].y * img_height
        position = x,y
        eyelid_position.append(position)
    return eyelid_position

def draw_facemesh(results , annotated_img):
    for face_landmarks in results.multi_face_landmarks:#基本的に一人しか検出しない設定
        mp_drawing.draw_landmarks(
              image=annotated_img,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_TESSELATION,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
              image=annotated_img,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_CONTOURS,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
              image=annotated_img,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_IRISES,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_iris_connections_style())
    return annotated_img

def draw_eyelid_right(face_landmarks,draw_img):
    landmark_subset = landmark_pb2.NormalizedLandmarkList(
          landmark = [
              face_landmarks.landmark[33],
              face_landmarks.landmark[246], 
              face_landmarks.landmark[161],
              
              face_landmarks.landmark[160], 
              face_landmarks.landmark[159],
              face_landmarks.landmark[158], 
              
              face_landmarks.landmark[157],
              face_landmarks.landmark[173],
              face_landmarks.landmark[133],
              
              face_landmarks.landmark[155], 
              face_landmarks.landmark[154],
              
              face_landmarks.landmark[153],
              face_landmarks.landmark[145],
              face_landmarks.landmark[144], 
              
              face_landmarks.landmark[163],
              face_landmarks.landmark[7],
#               face_landmarks.landmark[468],  #center
#               face_landmarks.landmark[470],  #upper
#               face_landmarks.landmark[472],  #below
          ]
    )
    mp_drawing.draw_landmarks(
        image=draw_img,
        landmark_list=landmark_subset)
    return draw_img

def draw_eyelid_left(face_landmarks,draw_img):
    landmark_subset = landmark_pb2.NormalizedLandmarkList(
          landmark = [
              face_landmarks.landmark[362],
              face_landmarks.landmark[398], 
              face_landmarks.landmark[384],
              
              face_landmarks.landmark[385], 
              face_landmarks.landmark[386],
              face_landmarks.landmark[387], 
              
              face_landmarks.landmark[388],
              face_landmarks.landmark[466],
              face_landmarks.landmark[263],
              
              face_landmarks.landmark[249], 
              face_landmarks.landmark[390],
              
              face_landmarks.landmark[373],
              face_landmarks.landmark[374],
              face_landmarks.landmark[380], 
              
              face_landmarks.landmark[381],
              face_landmarks.landmark[382],
          ]
    )
    mp_drawing.draw_landmarks(
        image=draw_img,
        landmark_list=landmark_subset)
    return draw_img

def draw_iris_right(face_landmarks,draw_img):
    img_height,img_width,_ = draw_img.shape
    j =468#虹彩の座標始まり(468～472)
    for i in range(1):
        x = face_landmarks.landmark[j].x * img_width
        y = face_landmarks.landmark[j].y * img_height
        cv2.circle(draw_img,center=(int(x), int(y)),radius=6,color=(255, 255, 255),thickness=-1)
        j+=1
    ##########
#         x_0 = face_landmarks.landmark[160].x * img_width#upper
#         y_0 = face_landmarks.landmark[160].y * img_height
#         x_1 = face_landmarks.landmark[153].x * img_width#below
#         y_1 = face_landmarks.landmark[153].y * img_height
#         p_0 = x_0,y_0
#         p_1 = x_1,y_1
#         radius_0 = distance.euclidean(p_0,p_1)/2
#         x_2 = face_landmarks.landmark[158].x * img_width#upper
#         y_2 = face_landmarks.landmark[158].y * img_height
#         x_3 = face_landmarks.landmark[144].x * img_width#below
#         y_3 = face_landmarks.landmark[144].y * img_height
#         p_2 = x_2,y_2
#         p_3 = x_3,y_3
#         radius_1 = distance.euclidean(p_2,p_3)/2
#         radius = (radius_0+radius_1)/2
    x_0 = face_landmarks.landmark[469].x * img_width#rightiris_left
    y_0 = face_landmarks.landmark[469].y * img_height
    x_1 = face_landmarks.landmark[471].x * img_width#rightirs_right
    y_1 = face_landmarks.landmark[471].y * img_height
    p_0 = x_0,y_0
    p_1 = x_1,y_1
    radius = distance.euclidean(p_0,p_1)/2
    ##########
    center_x = face_landmarks.landmark[468].x * img_width#center
    center_y = face_landmarks.landmark[468].y * img_height
    center = center_x,center_y
    ##########
    x_1 = face_landmarks.landmark[33].x * img_width#edge_left
    y_1 = face_landmarks.landmark[33].y * img_height
    x_2 = face_landmarks.landmark[133].x * img_width#edge_right
    y_2 = face_landmarks.landmark[133].y * img_height
    edge_0 = x_1,y_1
    edge_1 = x_2,y_2
    edge = distance.euclidean(edge_0,edge_1)
    trimming_width = edge * 1.5
    trimming_height = radius * 4
    size = trimming_height, trimming_width
    ##########
    return draw_img , radius , center , size

def trimming_eye(img , center , size):
    height = size[0]/2
    width = size[1]/2
    trimming_img = img[int(center[1] - height):int(center[1] + height),int(center[0] - width):int(center[0] + width)]
    return trimming_img