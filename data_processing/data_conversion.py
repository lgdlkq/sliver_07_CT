#! usr/bin/env python3
# coding = utf-8

import numpy as np
import SimpleITK as sitk
import os
import cv2

BASE_PATH = r"F:/sliver_07/"
SAVE_BASE_PATH = r'F:/sliver/'
IMG_BASE = BASE_PATH + 'volume-'
SEG_BASE = BASE_PATH + 'segmentation-'
seg = SAVE_BASE_PATH + 'seg/'
vol = SAVE_BASE_PATH + 'vol/'
neg_seg = SAVE_BASE_PATH + 'neg_seg/'
neg_vol = SAVE_BASE_PATH + 'neg_vol/'

# 创建文件夹
def createFile(filename):
    if os.path.exists(filename):
        pass
    else:
        os.makedirs(filename)

# 获取文件列表
def fileList(totlePath):
    pathDir = os.listdir(totlePath)
    return pathDir

# 加载nii文件
def loadFile(filename):
    ds = sitk.ReadImage(filename)
    img_array = sitk.GetArrayFromImage(ds)
    frame_num, width, height = img_array.shape
    # print(img_array.dtype)
    return img_array, frame_num, width, height

#获取文件夹文件数目
def geFileNum(filename):
    return len(fileList(filename))

def limitedEqualize(img_array, limit=2.0):
    img_array_list = []
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8,8))
    for img in img_array:
        img = img.astype(dtype=np.uint8)
        img_array_list.append(clahe.apply(img))
    return img_array_list

#png图片存储，文件大小很小，相比于jpg，做到了无损压缩，平均每张图大小增加10K左右，又比bmp具有压缩特性，平均大小比例为1:4
def save_png(seg_len):
    createFile(seg)
    createFile(vol)
    createFile(neg_seg)
    createFile(neg_vol)
    count = 0
    seg_fileNum = geFileNum(seg)
    neg_count = geFileNum(neg_seg)
    for i in range(seg_len):
        img_path = IMG_BASE + str(i) + '.nii'
        seg_path = SEG_BASE + str(i) + '.nii'
        vol_img, num, _, _ = loadFile(img_path)
        seg_img, num, _, _ = loadFile(seg_path)
        vol_img = limitedEqualize(vol_img, limit=4.)
        seg_img[seg_img == 2] = 1
        seg_img = seg_img.astype(dtype=np.float32) * 255.
        nn = neg_count-(i+1)*10
        neg_num = nn if nn > 0 else 0
        for j in range(num):
            if np.sum(seg_img[j] == 255) > 0 and count < seg_fileNum:
                count += 1
            elif count < seg_fileNum:
                continue
            elif nn==0:
                continue
            elif np.sum(seg_img[j] == 255) > 0:
                cv2.imwrite(seg + str(count) + '.png', seg_img[j])
                cv2.imwrite(vol + str(count) + '.png', vol_img[j])
                count += 1
            elif neg_num < 10:
                cv2.imwrite(neg_seg + str(neg_count) + '.png', seg_img[j])
                cv2.imwrite(neg_vol + str(neg_count) + '.png', vol_img[j])
                neg_num += 1
                neg_count +=1
        print(i,' seg: ',count,'   neg: ',neg_count)



if __name__ == '__main__':
    fls = fileList(BASE_PATH)
    all_len = len(fls)
    seg_len = all_len//2
    save_png(seg_len)
