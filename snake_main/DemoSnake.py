import numpy as np
import SeparabilitySnake as ss
import cv2
import math
from geomdl.fitting import approximate_curve

img = cv2.imread('test_img.png', cv2.IMREAD_GRAYSCALE)
#第1段階
h, w = img.shape
center = np.array([110, 110])#外郭を決める際の中心
radius = np.array([80, 40])#外郭を決める際の半径
theta = math.radians(45)#外郭を決める際の回転角
pts = ss.make_ellipse_points(center, radius, theta)#完全理解
pts = np.round(pts).astype(np.int32)#型の変換 , 小数点以下切り捨て
init_drawn = ss.draw_point_to_img(pts, img)#楕円上の点を結ぶ、あくまで表示する用に使う、第２段階では使わない
ss.show_cv_image(init_drawn)#plt.imshowと同じ
#第2段階
degree = 3
cs = 8
rect_size = np.array([30, 8])
curve = approximate_curve(pts.tolist(), degree, ctrlpts_size=cs)#Bスプライン補間、近似フィッティング
result_curve = ss.separability_snake(img, curve, rect_size, dif_abs=True)#領域間の分離度に基づく物体輪郭抽出
result_pts = np.array(result_curve.evalpts).astype(np.int32)
result_drawn = ss.draw_point_to_img(result_pts, img)
ss.show_cv_image(result_drawn)