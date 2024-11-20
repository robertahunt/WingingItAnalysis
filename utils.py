import cv2
import numpy as np

from scipy import ndimage

def get_biggest_contour(thres):
    contours,hierarchy = cv2.findContours(thres, 1, 2)
    area_sorted_indices = np.argsort([cv2.contourArea(x) for x in contours])
    biggest_contour_index = area_sorted_indices[-1]
    biggest_contour = contours[biggest_contour_index]
    return biggest_contour


def place_in_img(img, new, center_loc, ref_scale=1, rotation_angle=0, fill=0):
    if ref_scale != 1:
        new_shape = (np.array(new.shape[:2])*ref_scale).astype('int')
        new = cv2.resize(new, new_shape[::-1])
    if rotation_angle != 0:
        new = ndimage.rotate(new, rotation_angle, mode='constant', cval=fill)

    
    img_sx = center_loc[1] - (new.shape[1] // 2)
    left_crop = max(0 - img_sx, 0)
    
    img_ex = img_sx + new.shape[1]
    right_crop = min(img.shape[1] - img_ex, 0)

    img_sy = center_loc[0] - (new.shape[0] // 2)
    top_crop = max(0 - img_sy, 0)
    
    img_ey = img_sy + new.shape[0]
    bott_crop = min(img.shape[0] - img_ey, 0)
    
    img_sx = max(img_sx, 0)
    img_ex = min(img_ex, img.shape[1])
    img_sy = max(img_sy, 0)
    img_ey = min(img_ey, img.shape[0])

    if img_ex < 0:
        img_ex = 0
    if img_sx > img.shape[1]:
        img_sx = img.shape[1]
    if img_ey < 0:
        img_ey = 0
    if img_sy > img.shape[0]:
        img_sy = img.shape[0]

    img[img_sy:img_ey,img_sx:img_ex] = new[top_crop:new.shape[0]+bott_crop, left_crop:new.shape[1]+right_crop]
    return img.astype('uint8')

def is_wing_facing_up(img, mask):
    # decide if the wing is upside down based on vein locations
    mask_center = np.array(np.where(mask > 0.5)).mean(axis=1)
    
    seg = np.ones(img.shape)*255
    seg[np.where(mask > 0)] = img[np.where(mask>0)]
    veins = (seg[:,:,0] < 100).astype('uint8')*255
    
    #if DEBUG:
    #    plt.figure()
    #    plt.imshow(veins)
    #    plt.title('veins')
    veins_center = np.array(np.where(veins > 0.5)).mean(axis=1)
    
    if mask_center[0] < veins_center[0]:
        up = False
    else:
        up = True

    return up

def segment_contour(img, contour):
    """
    Given an image and a contour from that image, 
    return a segementation of the pixels inside the contour
        and the mask of the contour pixels
    """
    box_min_x, box_min_y = contour.min(axis=0)[0]
    box_max_x, box_max_y = contour.max(axis=0)[0]

    x,y,w,h = cv2.boundingRect(contour)
    
    seg = np.ones(img.shape)*255
    mask = np.zeros(img.shape)
    mask = cv2.drawContours(mask, [contour], -1, (1,1,1), -1)
    seg[np.where(mask > 0)] = img[np.where(mask>0)]
    
    return seg.astype('uint8'), mask