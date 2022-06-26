import numpy as np
import cv2 as cv
import math
import sys
sys.path.append("/work/Particle_Filter/circle_finder_main/circle_finder_main")
from circle_finder.csf import CircularSeparabilityFilter
from circle_finder.torch import EllipticalSeparabilityFilter

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

#パーティクルフィルタクラス
class ParticleFilter:
    #インスタンス宣言
    def __init__(self,size):
        self.size = size
        self.SAMPLEMAX = 1000 #最大サンプル数
        self.height, self.width = size[0],size[1]
    #パーティクル初期化
    def initialize(self):
        self.Y = np.random.random(self.SAMPLEMAX) * self.height#（0.0以上、1.0未満）の乱数×高さを返す。
        self.X = np.random.random(self.SAMPLEMAX) * self.width#（0.0以上、1.0未満）の乱数×幅を返す。
    #パーティクルの状態の更新　物体が適当な速さで適当に動くと仮定
    def modeling(self):#トリミングするサイズによって動く適度を変更する
        self.Y += np.random.random(self.SAMPLEMAX) * 20 - 10#元々は[*20-10]
        self.X += np.random.random(self.SAMPLEMAX) * 20 - 10
    #重み正規化
    def normalize(self, weight):
        return weight / np.sum(weight)
    #パーティクルリサンプリング , 重みに従ってパーティクルを選択 , 残ったパーティクルのインデックスを返す
    def resampling(self, weight):
        index = np.arange(self.SAMPLEMAX)#0〜SAMPLEMAXまでの配列を生成
        sample = []
        for i in range(self.SAMPLEMAX):#SAMPLEMAX個のサイズがweightのindexを生死絵する
            idx = np.random.choice(index, p=weight)#pパラメータを使うことで各要素の出る確率が異なる。つまり、重みの大きさにしたがってidx()生成！！
            sample.append(idx)
        return sample     
    #尤度計算
    def calcLikelihood(self, img, radius, width,height):
        intensity = []
        x_in=int(radius)
        y_in=int(radius)
        x_out=int(radius+10)
        y_out=int(radius+10)
        axes_in = x_in,y_in
        axes_out = x_out,y_out
        epmap,cut_img,circles = a(img,x_in,y_in,x_out,y_out) #epmapは尤度画像、cut_imgは尤度の高い周辺画像（BGR）
        #場外のサンプルに対しての処理
        for i in range(self.SAMPLEMAX):
            y = int(self.Y[i])
            x = int(self.X[i])
            if y >= 0 and y < height-1 and x >= 0 and x < width-1:
                intensity.append(epmap[y,x])
            else:
                intensity.append(-1)
        weights = intensity#リスト型
        weights[weights == -1] = 0
        #重みに負の値がある場合の処理
        flag=True
        while flag==True:#weights(リスト)の負の要素を0にする
            x =min(weights)
            idx=weights.index(x)
            if x<0:
                weights[idx]=0
            else:
                flag=False
        #正規化
        weights = self.normalize(weights)
        return weights
    #期待値を返す
    def filtering(self, image, radius ,width,height):
        self.modeling()#全体にサンプルをばらまく
        weights = self.calcLikelihood(image,radius,width,height)#尤度計算
        index = self.resampling(weights)#重みを確率としてリサンプル , 重み確率の高いものが添字として抽出されやすい
        self.Y = self.Y[index]
        self.X = self.X[index]
        return np.sum(self.Y) / float(len(self.Y)), np.sum(self.X) / float(len(self.X))
    def plot(self, x_after, y_after, x_before, y_before, center_eye, draw_img):
        center_eye_x,center_eye_y = center_eye[0],center_eye[1]
        alpha = x_after - x_before
        beta = y_after - y_before
        center_eye_x_1 = center_eye_x+alpha
        center_eye_y_1 = center_eye_y+beta
        center_eye = (int(center_eye_x_1), int(center_eye_y_1))
        draw_img = cv.circle(draw_img,center_eye,radius=5,color=(255, 255, 255),thickness=-1)
        return draw_img, center_eye
            
            
            
            
            
            