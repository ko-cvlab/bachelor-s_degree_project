import numpy as np
import cv2
import math
from geomdl.visualization import VisMPL as vis #オブジェクト指向のBスプラインおよびNURBSライブラリである。
from geomdl.fitting import approximate_curve #フィッティング法、曲線近似に使う

def add_margin(inside, margin):#関数3_1
    img_shape = tuple([s + 2*margin for s in inside.shape])#sは縦と横のサイズ、inside.shapeはタプル型、marginは境界線上に加算するから2*margin
    img = np.zeros(img_shape,dtype=inside.dtype)#inside.dtype：対象画像の型
    img += np.min(inside)#対象画素中の最小値を余白(zeros)に足す
    img[margin:img_shape[0]-margin, margin:img_shape[1]-margin] += inside - np.min(inside)#対象画像に対してinside-np.min(inside)の差分だけ足す
    return img

def evalpts_u(curve, div):#関数3_2,div=40
    range_u = np.linspace(0, 1, div, endpoint=False)#Falseより、最後に１を含まない
    return np.array(curve.evaluate_list(range_u))#approximate_curveに含まれる変数,Evaluate the curve for an input range of parameters
    
def cul_normal_vector(center, neighborhood):#関数3_3_1_1
    p_vec = (neighborhood[1] - neighborhood[0])
    p_norm = np.linalg.norm(p_vec)#L2ノルム(ユークリッド距離)を求めている、ord[次元]はこの場合は２だと思う。
    if p_norm != 0 : p_vec = p_vec / p_norm
    else : p_vec = [0, 0]
    v_vec = np.array([p_vec[1], -p_vec[0]])#法線ベクトル（正規化）,p_vecとの内積は0
    return v_vec

def cul_separability(clipped_image):#関数3_3_1_3
    h, w = clipped_image.shape
    N = w*h
    variance_all = N * np.var(clipped_image)#分散を求める
    average_all = np.mean(clipped_image)#平均を求める
    variance_boundary = [(i+1)*h*(np.mean(clipped_image[:,:i+1])-average_all) ** 2
                         + (w-i-1)*h*(np.mean(clipped_image[:,i+1:])-average_all) ** 2 for i in range(w-1)]
    separability = variance_boundary/variance_all
    return (np.max(separability), np.argmax(separability)) if variance_all != 0 else (0, int(w/2))#分散が０であれば、ｗの半分を境界点とする

def clip_rect(img, center, rect_size, direction):#関数3_3_1_2、direction=normal(ユークリッド距離を用いたやつ)
    simg_size = rect_size[0] + 10 #rect_size[0]=30
    start = np.array([int(center[i])-int(simg_size/2) for i in range(2)])
    end = np.array([int(center[i])+int(simg_size/2) + (1 if simg_size % 2 == 1 else 0) for i in range(2)])#simg_sizeが小さい場合は１にする
    rotated_img = img[start[1]:end[1], start[0]:end[0]].copy()#画素値を逆にする？
    direction_norm = np.linalg.norm(direction)#例：4/5 , 3/5 の場合には、三平方の定理より求まる。
    if direction_norm != 0 : direction = direction / direction_norm
    else : direction = np.zeros(3)
    rotation_angle = math.atan2(direction[1], direction[0])#arctan(y/x)をラジアンで返す,(ex)atan2(1,-1)=135
    M = cv2.getRotationMatrix2D((simg_size/2,simg_size/2), np.degrees(rotation_angle),1)#回転の変換行列を生成,第一引数が回転の原点となる座標、第二引数が回転の角度（ラジアンではなく度degree）、第三引数が拡大、縮小倍率
    rotated_img = cv2.warpAffine(rotated_img, M, (simg_size, simg_size), flags=cv2.INTER_NEAREST)#アフィン変換を実行,INTER_NEAREST:最近傍補間
    start  = np.array([int(simg_size/2)-int(rect_size[i]/2) for i in range(2)])
    end = np.array([int(simg_size/2)+int(rect_size[i]/2) + (1 if rect_size[i] % 2 == 1 else 0) for i in range(2)])
    clipped_img = rotated_img[start[1]:end[1], start[0]:end[0]]
    return clipped_img

def update_sample_point(center, neighborhood, image, rect_size, w):#関数3_3_1
    normal = cul_normal_vector(center, neighborhood)#関数3_3_1_1
    clipped_image = clip_rect(image,center,rect_size,normal)#関数3_3_1_2、rect_size = np.array([30, 8])
    
    separability, boundary = cul_separability(clipped_image)#関数3_3_1_3
    add = np.mean(clipped_image[:,:boundary+1]) < np.mean(clipped_image[:,boundary+1:])# どういう意味？
    return separability, center + (boundary + add - int(rect_size[0]/2)) * normal * w

def update_evalpts(evalpts,img,rect_size,div,w=1.0):#関数3_3、div=40,w=0.95(デフォルト値:1,引数がなかった場合に使われる-)
    sep_pts = np.array([update_sample_point(#関数3_3_1
        evalpts[(u+1)%div],np.array([evalpts[u],evalpts[(u+2)%div]]),img,rect_size,w) for u in range(div)])
    return np.mean(sep_pts[:,0]), np.array([sep_pts[i,1] for i in range(div)])

def separability_snake(img, init, rect_size, div=40, N_limit=20, dif_limit=0.001, dif_abs=False, increasing_limit=0.01, ctrlpts_increasing=True, c=0.95, ctrlpts_size=8, w=0.95, debug=True, max_ctrlpts_size=20):#関数3
    margin = rect_size[0]#rect_size = np.array([30, 8])
    img = add_margin(img, margin)#関数3_1（余白追加）
    newcp = np.array(init.ctrlpts) + margin #余白を追加したことにより制御点の位置を更新する
    init.ctrlpts = newcp.tolist()
    prev_separability = 0
    for _ in range(N_limit):
        evalpts = evalpts_u(init,div)#関数3_2、？？？
        mean_separability, updated_evalpts = update_evalpts(evalpts,img,rect_size.astype(np.int64),div,w)#関数3_3,div=40,w=0.95
        if debug:print(f"mean_separability:{mean_separability}")#各制御点においての平均分離度
        if dif_abs :#全サンプル点におけるηの平均の変化が基準εより小さい場合
            if math.fabs(mean_separability - prev_separability) < dif_limit : break
        else:#全サンプル点におけるηの平均の変化が基準εより小さい場合
            if mean_separability - prev_separability < dif_limit : break
        init = approximate_curve(updated_evalpts.tolist(),3,ctrlpts_size = ctrlpts_size)
        if ctrlpts_increasing:
            if math.fabs(mean_separability - prev_separability) < increasing_limit and ctrlpts_size < max_ctrlpts_size:#制御点追加アルゴリズム
                ctrlpts_size = ctrlpts_size + 1#制御手が加えられるのは理解だけど、場所は適当でいいの？
        if debug:print(f"ctrlpts_size:{ctrlpts_size}")
        prev_separability = mean_separability
        if debug:print(f"rect_size.astype(np.int64):{rect_size.astype(np.int64)}")
        rect_size = np.array([rs * c if 1 <= rs * c else 1 for rs in rect_size]) #rect_size = np.array([30, 8]),c=0.95
    newcp = np.array(init.ctrlpts) - margin #余白を引いて元に戻す
    init.ctrlpts = newcp.tolist()
    return init

def draw_point_to_img(pts, img):#関数2
    img_c = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)#グレー画像からカラー画像を生成
    drawn = cv2.polylines(img_c, [pts], True, (0, 0, 255), thickness=1)#楕円上の点を結ぶ
    return drawn
def make_ellipse_points(center,radius,theta,psize=40):#関数1
    points = np.array([[radius[0]*math.cos(u), radius[1]*math.sin(u)] for u in np.linspace(0,2*math.pi,num=psize, endpoint=False)]).T#楕円上の点
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta),  math.cos(theta)]])#回転行列
    points = np.dot(rotation, points).T#アフィン変換（拡大縮小、回転、平行移動）
    points += center
    return points
def show_cv_image(img):#関数4
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()