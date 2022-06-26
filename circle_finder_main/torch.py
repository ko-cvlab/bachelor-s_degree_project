__all__ = ['expand_dim_to_3', 'parametric_ellipse', 'elliplise', 'EllipticalSeparabilityFilter']

# Cell
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
from torchvision.transforms import ToTensor
from skimage.feature import peak_local_max
from .core import *

# Cell
def expand_dim_to_3(arr):
    if arr.ndim == 2:
        return np.expand_dims(arr, axis=-1)
    elif arr.ndim == 3:
        return arr
    else:
        raise ValueError()

# Cell

def parametric_ellipse(alpha, A, a, b):#完全に理解
    X = a * np.cos(alpha) * np.cos(A) - b * np.sin(alpha) * np.sin(A)#楕円、余弦の加法定理、cos(α+A)
    Y = a * np.cos(alpha) * np.sin(A) + b * np.sin(alpha) * np.cos(A)#楕円、正弦の加法定理、sin(α+A)
    return X, Y#楕円上の座標を取得している

def elliplise(axes, angle, center=None, width=None, height=None):#完全に理解
    a = axes[0]
    b = axes[1]
    A = np.deg2rad(angle)#ラジアンを度に直す
    cc = [parametric_ellipse(alpha, A, a, b) for alpha in np.linspace(0, np.pi*2, 1000)]   #内向表記
    #第三引数numに要素数を指定する。それらに応じた間隔（公差）が自動的に算出される。つまり、２π(360度)1000個に分ける
    #楕円上の座標を取得している？
    if height is None and width is None:
        width, height = np.max(cc, axis=0)#座標上の最大値をとる
        width = round(width) * 2 + 1 #+1するので実際のサイズは、(2*半径+1,2*半径+1)になる
        height = round(height) * 2 + 1
    if center is None:
        center = (round(width/2), round(height/2))
    mask = np.zeros((height, width), np.uint8)#マスク画像の生成、真っ黒画像
    template = torch.Tensor(cv.ellipse(mask, center, axes, angle, color=1, thickness=-1, startAngle=0, endAngle=360)).unsqueeze(0)
    return template


class EllipticalSeparabilityFilter:#完全に理解
    def __init__(self, axes_in, axes_out, angle):#完全に理解
        self.axes_in = axes_in
        self.axes_out = axes_out
        self.angle = angle
        inner_region, outer_region, full_region = self.ellipse_templates()#ここで内側、外側、全体の領域範囲を求める
        self.inner_region = inner_region
        self.outer_region = outer_region
        self.full_region = full_region

    def __call__(self, img):#完全に理解（難易度高い）
        axes_in, axes_out, angle = self.axes_in, self.axes_out, self.angle
        inner_region, outer_region, full_region = self.inner_region, self.outer_region, self.full_region
        #print(inner_region.shape) #torch.Size([1, 61, 61])
        #print(outer_region.shape) #torch.Size([1, 61, 61])
        #print(full_region.shape) #torch.Size([1, 61, 61])
            
        top = bottom = axes_out[0]
        right = left = axes_out[1]
        
        #エッジのピクセルをコピー , エッジのカラーを受け継ぐイメージ(この場合、真っ黒？、そのために+1したんでしょ？)
        borderType = cv.BORDER_REPLICATE
        img = cv.copyMakeBorder(img, top, bottom, left, right, borderType)#画像の縁部分に対して拡張
        img = ToTensor()(img)
        #print(img.shape) #torch.Size([3, 168, 252])
        c = img.shape[0]#カラーチャンネル数を取得
        #print(c) #3
        
        #元のテンソルを書き換えずに、次元を増やしたテンソルを返す,
        img = img.unsqueeze(0)   
        #print(img.shape)   #torch.Size([1, 3, 168, 252])
        
        n_inner = inner_region.sum()#画素の合計を求める
        n_outer = outer_region.sum()
        n_full = n_inner + n_outer
        #print(n_inner) #tensor(1307.)
        #print(n_outer) #tensor(1608.)
        #print(n_full) #tensor(2915.)
        
        #broadcasting用にarrayの形を整えてる,#repeat(要素や配列, 繰り返し回数 ,(繰り返す方向) )
        #イメージ：奥、縦、横に何個用意するかってこと！
        w_inner = inner_region.repeat([c,1,1,1]) / n_inner #内側フィルタ
        w_outer = outer_region.repeat([c,1,1,1]) / n_outer #ドーナツ型フィルタ
        w_full = full_region.repeat([c,1,1,1]) / n_full #内側＋ドーナツ型フィルタ
        #print((w_inner))   
        #print(w_inner.shape)   #torch.Size([3, 1, 61, 61])
        #print(w_outer)  
        #print(w_outer.shape)   #torch.Size([3, 1, 61, 61])
        #print(w_full)   
        #print(w_full.shape)   #torch.Size([3, 1, 61, 61])
             
        #畳み込みで各領域の平均を計算 , この時、可能であるすべての座標を楕円(カーネル)の中心として、"平均値"を求める
        #CNNの畳み込みはブロックの各ピクセルに対して重みをかけて合計するって処理←「ブロックに分ける」と「重みをかけて合計する」の二つの処理をする
        m_inner = torch.nn.functional.conv2d(img, w_inner, groups=c)[0]
        m_outer = torch.nn.functional.conv2d(img, w_outer, groups=c)[0]
        m_full = torch.nn.functional.conv2d(img, w_full, groups=c)[0]
        #print(m_inner.shape)   #torch.Size([3, 108, 192])
        
        #クラス間分散マップを計算 , #ω1(m1-mt)^2+ω2(m2-mt)^2：クラス間分散の分子を計算
        sb_map = n_inner * (m_inner - m_full)**2 + n_outer * (m_outer - m_full)**2

        #unfoldは「ブロックに分ける」処理をしてくれるイメージ
  
        #permute:次元の入れ替え , unfold:バッチ化された入力テンソルからスライディングローカルブロックを抽出します。
        unfolded = torch.nn.functional.unfold(img.permute((1,0,2,3)), w_full.shape[-2:])  #クラス内分散は、内側、外側一緒に分割するイメージ！
        #print(unfolded.shape)   #torch.Size([3, 3721, 20736])
        meansubbed = unfolded - m_full.reshape(c, 1, -1)#元の次元数より-1する
        #print(meansubbed.shape)   #torch.Size([3, 3721, 20736])
        squared = meansubbed ** 2
        st_map = (w_full.reshape(c, 1, -1) @ squared).reshape(sb_map.shape[-3:])#st_mapは「重みをかけて合計する」じゃあ計算できない
        out = sb_map / st_map
        return out.mean(axis=0, keepdim=True)

    def ellipse_templates(self):#完全に理解
        axes_in, axes_out, angle = self.axes_in, self.axes_out, self.angle
        full_region = elliplise(axes_out, angle)#楕円形のテンプレート画像を取得（外半径を基準にしている）
        height, width = full_region.shape[1:]
        #楕円形のテンプレート画像を取得（内半径を基準にしている） ,高さと幅は、外半径のものを採用している！この場合だと、(61,61)になる！！
        inner_region = elliplise(axes=axes_in, angle=angle, height=height, width=width)
        outer_region = full_region - inner_region#ドーナツ型のテンプレート画像を取得
        return inner_region, outer_region, full_region
 
    def find_circles(self, img, num_circles=None):
        epmap = self.__call__(img)
        epmap = epmap.to('cpu').detach().numpy().copy()
        epmap = np.squeeze(epmap)
        epmap[np.isnan(epmap)]=0
        peaks = peak_local_max(epmap)

        if num_circles is None:
            return peaks
        else:
            return epmap,peaks[:num_circles]

        
        
        
        
        
        
    def cut_img_2(self, img, circles):#カットできない画像は省く、学習画像生成用
        num = len(circles)
        n=0
        datas=[]
        for n in range(num):
            x_circles = circles[n]
            cx= x_circles[0]
            cy= x_circles[1]
            data = img[int(cx-30):int(cx+30) , int(cy-30):int(cy+30)]#size=(60,60)にさせる！            
            if (data.shape[0]==60 and data.shape[1]==60):
                data_expanded = np.expand_dims(data,axis=0)
                datas.append(data_expanded)
            else:
                 continue
        image_datas = np.concatenate(datas,axis=0)
        
        return image_datas
    
    def cut_img_3(self, img, circles):#カットできない画像には黒画像を返す
        num = len(circles)
        a = self.axes_out[0]
        b = self.axes_out[0]
        n=0
        datas=[]
        empty_data = np.zeros((60,60,3))
        empty_data = cv.cvtColor(empty_data.astype("uint8"), cv.COLOR_RGB2HSV)
        for n in range(num):
            x_circles = circles[n]
            cx= x_circles[0]
            cy= x_circles[1]
            data = img[int(cx-1.5*b):int(cx+1.5*b) , int(cy-1.5*b):int(cy+1.5*b)]#size=(60,60)になるはず
            if (data.shape[0]==60 and data.shape[1]==60):
                #60という値は、カットする画像の大きさによって変わる
                data_expanded = np.expand_dims(data,axis=0)
                datas.append(data_expanded)
            else:
                data_expanded = np.expand_dims(empty_data,axis=0)
                datas.append (data_expanded)
                
        # (n_samples,height,width,channels)
        image_datas = np.concatenate(datas,axis=0)
        
        return image_datas
    
    def cut_img_4(self, img, circles):#カットできない画像には黒画像を返す
        num = len(circles)
        #a = self.axes_out[0]
        #b = self.axes_out[0]
        n=0
        datas=[]
        empty_data = np.zeros((60,60,3))
        empty_data = cv.cvtColor(empty_data.astype("uint8"), cv.COLOR_RGB2HSV)
        for n in range(num):
            x_circles = circles[n]
            cx= x_circles[0]
            cy= x_circles[1]
            data = img[int(cx-30):int(cx+30) , int(cy-30):int(cy+30)]#size=(60,60)にさせる！            
            if (data.shape[0]==60 and data.shape[1]==60):
                data_expanded = np.expand_dims(data,axis=0)
                datas.append(data_expanded)
            else:
                data_expanded = np.expand_dims(empty_data,axis=0)
                datas.append (data_expanded)
                
        # (n_samples,height,width,channels)
        image_datas = np.concatenate(datas,axis=0)
        
        return image_datas
    
    def cut_img_5(self, img, circles):#40,60cm用
        num = len(circles)
        a = self.axes_out[0]
        b = self.axes_out[1]
        n=0
        datas=[]
        empty_data = np.zeros((3*a,3*b,3))
        empty_data = cv.cvtColor(empty_data.astype("uint8"), cv.COLOR_RGB2HSV)
        for n in range(num):
            x_circles = circles[n]
            cx= x_circles[0]
            cy= x_circles[1]
            data = img[int(cx-1.5*b):int(cx+1.5*b) , int(cy-1.5*a):int(cy+1.5*a)] 
            if (data.shape[0]==3*a and data.shape[1]==3*b):
                data_expanded = np.expand_dims(data,axis=0)
                datas.append(data_expanded)
            else:
                data_expanded = np.expand_dims(empty_data,axis=0)
                datas.append (data_expanded)
                
        # (n_samples,height,width,channels)
        image_datas = np.concatenate(datas,axis=0)
        
        return image_datas