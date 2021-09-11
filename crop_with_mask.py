#-*-coding:utf-8-*-

import os
import cv2
import numpy as np
from PIL import Image
import math
import time
from config import *

crop_count = 400
crop_size = 512

rd_degrees = [i for i in range(0, 89, 9)]

def crop(img, x, y, angle):
    print(x, y, angle)

    h, w = img.shape
    M = cv2.getRotationMatrix2D((y, x), angle, 1)
    rotated_t = cv2.warpAffine(img.astype(np.float), M, (w, h))
    rotated_t = rotated_t[x-crop_size//2:x+crop_size//2, y-crop_size//2:y+crop_size//2]
    return rotated_t


def process():
    for file in os.listdir(tiff_mask_fd):
        if not file.endswith('png'):
            continue
        print(file)
        f, suf = file.split('.')
        os.makedirs(os.path.join(terrain_png_fd, f), exist_ok=True)
        im = Image.open(os.path.join(tiff_data_fd, f + '.tif'))
        imarray = np.array(im)
        mask_orig = cv2.imread(os.path.join(tiff_mask_fd, file), 0)

        imarray_with_mask = []
        mask_erosions = []
        for degree in rd_degrees:
            border_size = crop_size//2 + math.ceil(crop_size / 2 * math.sin(math.radians(degree)))
            kernel = np.ones((2 * border_size, 2 * border_size), np.int)        #腐蚀的kernel需要乘以2才是我想要的腐蚀大小
            mask_erosion = cv2.erode(mask_orig, kernel,  borderType=cv2.BORDER_CONSTANT, borderValue=0)
            mask_erosions.append(mask_erosion)

            tp = np.copy(imarray);tp[mask_erosions==0] = 0
            imarray_with_mask.append(tp)

        for i in range(crop_count):
            save_name = os.path.join(terrain_png_fd, f, f + '_%d.png' % (i))
            if os.path.exists(save_name):
                continue

            print(f, i)
            rd_angle_idx = np.random.randint(0, len(rd_degrees))
            valid_centers = np.where(mask_erosions[rd_angle_idx]>230)

            rd_center_idx = np.random.randint(0, len(valid_centers[0]))

            rd_img = crop(imarray_with_mask[rd_angle_idx], valid_centers[0][rd_center_idx], valid_centers[1][rd_center_idx], rd_degrees[rd_angle_idx])
            rd_img = (rd_img - np.min(rd_img)) / (np.max(rd_img) - np.min(rd_img)) * 65535
            Image.fromarray(rd_img.astype(np.uint16)).save(save_name)
            #print(rd_img.shape)


if __name__ == '__main__':
    process()

