import numpy as np
import cv2
import math
from geomdl.visualization import VisMPL as vis
from geomdl.fitting import approximate_curve

def add_margin(inside, margin):
    img_shape = tuple([s + 2*margin for s in inside.shape])

    img = np.zeros(img_shape,dtype=inside.dtype)
    img += np.min(inside)
    img[margin:img_shape[0]-margin, margin:img_shape[1]-margin] += inside - np.min(inside)

    return img

def evalpts_u(curve, div):
    range_u = np.linspace(0, 1, div, endpoint=False)
    return np.array(curve.evaluate_list(range_u))

def cul_normal_vector(center, neighborhood):
    p_vec = (neighborhood[1] - neighborhood[0])
    p_norm = np.linalg.norm(p_vec)
    if p_norm != 0:    
        p_vec = p_vec / p_norm
    else:
        p_vec = [0, 0]
    v_vec = np.array([p_vec[1], -p_vec[0]])

    return v_vec

def cul_separability(clipped_image):
    h, w = clipped_image.shape
    N = w*h

    variance_all = N * np.var(clipped_image)
    average_all = np.mean(clipped_image)
    variance_boundary = [(i+1)*h*(np.mean(clipped_image[:,:i+1])-average_all) ** 2
                   + (w-i-1)*h*(np.mean(clipped_image[:,i+1:])-average_all) ** 2
                    for i in range(w-1)]
    
    separability = variance_boundary/variance_all

    return (np.max(separability), np.argmax(separability)) if variance_all != 0 else (0, int(w/2))

def clip_rect(img, center, rect_size, direction):
    simg_size = rect_size[0] + 10
    start = np.array([int(center[i])-int(simg_size/2) for i in range(2)])
    end = np.array([int(center[i])+int(simg_size/2) + (1 if simg_size % 2 == 1 else 0) for i in range(2)])
    
    rotated_img = img[start[1]:end[1], start[0]:end[0]].copy()

    direction_norm = np.linalg.norm(direction)
    if direction_norm != 0:
        direction = direction / direction_norm
    else:
        direction = np.zeros(3)

    rotation_angle = math.atan2(direction[1], direction[0])
    
    M = cv2.getRotationMatrix2D((simg_size/2,simg_size/2), np.degrees(rotation_angle),1)
    
    rotated_img = cv2.warpAffine(rotated_img, M, (simg_size, simg_size), flags=cv2.INTER_NEAREST)
   
    start  = np.array([int(simg_size/2)-int(rect_size[i]/2) for i in range(2)])
    end = np.array([int(simg_size/2)+int(rect_size[i]/2) + (1 if rect_size[i] % 2 == 1 else 0) for i in range(2)])

    clipped_img = rotated_img[start[1]:end[1], start[0]:end[0]]

    return clipped_img

def update_sample_point(center, neighborhood, image, rect_size, w):
    normal = cul_normal_vector(center, neighborhood)
    clipped_image = clip_rect(image,center,rect_size,normal)
    separability, boundary = cul_separability(clipped_image)

    add = np.mean(clipped_image[:,:boundary+1]) < np.mean(clipped_image[:,boundary+1:])

    return separability, center + (boundary + add - int(rect_size[0]/2)) * normal * w

def update_evalpts(evalpts,img,rect_size,div,w=1.0):
    sep_pts = np.array([update_sample_point(evalpts[(u+1)%div],
            np.array([evalpts[u], evalpts[(u+2)%div]]),
            img,rect_size,w) for u in range(div)])

    return np.mean(sep_pts[:,0]), np.array([sep_pts[i,1] for i in range(div)])

def separability_snake(img, init, rect_size, div=40, N_limit=20, dif_limit=0.001, dif_abs=False,
                          increasing_limit=0.01, ctrlpts_increasing=True, c=0.95, ctrlpts_size=8,
                          w=0.95, debug=True, max_ctrlpts_size=20):
    margin = rect_size[0]
    img = add_margin(img, margin)

    newcp = np.array(init.ctrlpts) + margin
    init.ctrlpts = newcp.tolist()
    
    prev_separability = 0
    for _ in range(N_limit):
        evalpts = evalpts_u(init,div)
        mean_separability, updated_evalpts = update_evalpts(evalpts,img,rect_size.astype(np.int64),div,w)
  
        if debug:
            print(mean_separability)

        if dif_abs :
            if math.fabs(mean_separability - prev_separability) < dif_limit:
                break
        else:
            if mean_separability - prev_separability < dif_limit:
                break

        init = approximate_curve(updated_evalpts.tolist(),3,ctrlpts_size = ctrlpts_size)

        if ctrlpts_increasing:
            if math.fabs(mean_separability - prev_separability) < increasing_limit and ctrlpts_size < max_ctrlpts_size:
                ctrlpts_size = ctrlpts_size + 1
        
        if debug:
            print(ctrlpts_size)

        prev_separability = mean_separability

        if debug:
            print(rect_size.astype(np.int64))

        rect_size = np.array([rs * c if 1 <= rs * c else 1 for rs in rect_size])

    newcp = np.array(init.ctrlpts) - margin
    init.ctrlpts = newcp.tolist()
        
    return init

def draw_point_to_img(pts, img):
    img_c = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    drawn = cv2.polylines(img_c, [pts], True, (0, 0, 255), thickness=1)
    return drawn

def make_ellipse_points(center,radius,theta,psize=40):
    points = np.array([[radius[0]*math.cos(u), radius[1]*math.sin(u)]
                    for u in np.linspace(0,2*math.pi,num=psize, endpoint=False)]).T
    rotation = np.array([[math.cos(theta), -math.sin(theta)],
                         [math.sin(theta),  math.cos(theta)]])
    points = np.dot(rotation, points).T
    points += center
    return points

def show_cv_image(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()