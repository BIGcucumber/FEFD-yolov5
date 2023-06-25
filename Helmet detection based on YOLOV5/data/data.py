import os
import random
import shutil
from functools import reduce

imagefilepath = '/home/bai/Bai/qiuyi/datasets/sixdetect/images'
labelfilepath = '/home/bai/Bai/qiuyi/datasets/sixdetect/txt'
path_image={
    'image_train':"./sixdetect/images/train",
    'image_val' : "./sixdetect/images/val",
    'image_test' : "./sixdetect/images/test",
    'label_train' :'./sixdetect/labels/train',
    'label_val' : './sixdetect/labels/val',
    'label_test' : './sixdetect/labels/test',
}

def mkdir(path):
    # path = path.strip()
    # path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
    else:
        print(path + ' 目录已存在')

def remore(form,file):
    '''
    移动图片
    '''
    imagesrc = os.path.join(imagefilepath, file+'.jpg')
    imagedst = os.path.join(path_image['image_'+form], file+'.jpg')
    labelsrc = os.path.join(labelfilepath, file + '.txt')
    labeldst = os.path.join(path_image['label_'+form], file + '.txt')
    shutil.copyfile(imagesrc, imagedst)
    shutil.copyfile(labelsrc, labeldst)
#
# for key in path_image:
#     #路径存在则退出，不存在则创建
#     mkdir(path_image[key])
# trainval_percent = 0.9  # 训练和验证集所占比例，剩下的0.1就是测试集的比例
# train_percent = 0.8  # 训练集所占比例，可自己进行调整
# total_xml = os.listdir(labelfilepath)
# num = len(total_xml)
# list = range(num)
# tv = int(num * trainval_percent)
# tr = int(tv * train_percent)
# trainval = random.sample(list, tv)
# train = random.sample(trainval, tr)
#
# for i in list:
#     name = total_xml[i][:-4]
#     if i in trainval:
#         if i in train:
#             remore('train',name)
#         else:
#             remore('val',name)
#     else:
#         remore('test',name)
import cv2
import numpy as np
def splicing_img(img_file1,img_file2,out_file,tunnel,border_position,border_width):
    print('file1=' + img_file1 + ', file2=' + img_file2)
    img1 = cv2.imread(img_file1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_file2, cv2.IMREAD_GRAYSCALE)
    #第二个参数为如何读取图片，包括cv2.IMREAD_COLOR：读入一副彩色图片；cv2.IMREAD_GRAYSCALE：以灰度模式读入图片；cv2.IMREAD_UNCHANGED：读入一幅图片，并包括其alpha通道。
    hight,width = img1.shape
    final_matrix = np.zeros((hight,width), np.uint8) #,tunnel), np.uint8) #高*款（y，x）20*20*1
    # change
    x1=0
    y1=hight
    x2=width
    y2=0   #图片高度，坐标起点从上到下
    if border_position =='top':
        final_matrix[y2 + border_width:y1, x1:x2] = img1[y2:y1 - border_width, x1:x2]
        final_matrix[y2:border_width, x1:x2] = img2[y2:border_width, x1:x2]
    #左侧增加边或空白
    if border_position == 'left':
        final_matrix[y2 :y1, x1+ border_width:x2] = img1[y2:y1, x1:x2 - border_width]
        final_matrix[y2:y1, x1:border_width] = img2[y2:y1, x1:border_width]

    if border_position == 'right':
        final_matrix[y2 :y1, x1:x2 - border_width] = img1[y2:y1, x1 + border_width:x2]
        final_matrix[y2:y1, x2-border_width:x2] = img2[y2:y1, x1:border_width]
    #底部增加边或空白
    if border_position =='bottom':
        final_matrix[y2 :y1 - border_width, x1:x2] = img1[y2+ border_width:y1 , x1:x2]
        final_matrix[y1 - border_width:y1, x1:x2] = img2[y2:border_width, x1:x2]
    if border_position =='copy':
        final_matrix[y2 :y1, x1:x2] = img1[y2:y1 , x1:x2]

    cv2.imwrite(out_file, final_matrix)

    return final_matrix


def rotationImg(img_file1, out_file, ra):
    # 获取图片尺寸并计算图片中心点
    img = cv2.imread(img_file1, cv2.IMREAD_GRAYSCALE)
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, ra, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    # cv2.imshow("rotated", rotated)
    # cv2.waitKey(0)
    cv2.imwrite(out_file, rotated)

    return rotated


def cut_img(img_file1, out_file, top_off, left_off, right_off, bottom_off):
    img1 = cv2.imread(img_file1, cv2.IMREAD_GRAYSCALE)
    hight, width = img1.shape
    x1 = 0
    y1 = hight
    x2 = width
    y2 = 0  # 图片高度，坐标起点从上到下hight,width = img1.shape

    # 灰度图像，不使用通道tunnel
    final_matrix = np.zeros((hight, width), np.uint8)  # ,tunnel), np.uint8) #高*款（y，x）20*20*1
    final_matrix[y2 + top_off:y1 - bottom_off, x1 + left_off:x2 - right_off] = img1[y2 + top_off:y1 - bottom_off,
                                                                               x1 + left_off:x2 - right_off]

    cv2.imwrite(out_file, final_matrix)

    return final_matrix

def sp_noiseImg(img_file1,prob):
    image = cv2.imread(img_file1, cv2.IMREAD_GRAYSCALE)
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
if __name__ == '__main__':
    pass

# isExists = os.path.exists(imagefilepath)
# print(isExists)in
# import torch
# print(torch.version.cuda)
# print(torch.cuda.is_available())
# from tensorboard import version
# print(version.VERSION)
# filelist=os.listdir(labelfilepath)
# for files in filelist:
#     Olddir=os.path.join(labelfilepath,files)
#     if os.path.isdir(Olddir):
#         continue
#     Newdir=os.path.join(labelfilepath,files[4:])
#     os.rename(Olddir,Newdir)
#     # print(Olddir,Newdir)
#
def get_bonus(num):
    if(num<=10):
        return num*0.05
    if(num<=20):
        return 10*0.05+(num-10)*0.04
    if(num<=40):
        return 10*0.05+10*0.04+(num-20)*0.05
    if(num<=60):
        return 10*0.05+10*0.04+20*0.05+(num-40)*0.03
    if(num<=100):
        return 10*0.05+10*0.04+20*0.05+40*0.03+(num-60)*0.015
    else:
        return 10*0.05+10*0.04+20*0.05+40*0.03+40*0.015+(num-100)*0.01
a= input()
P=input().split()
Q=input().split()
dic_num={}
result=[]
results=[]
for i in P:
    dic_num[i]=dic_num.get(i,0)+1
    if i not in Q:
        results.append(i)
for i in Q:
    n=dic_num[i]
    result.extend([i]*n)
'''
1
1 7 765 1  7675 54 766 7567 7 757 7675 67567
765 7 
'''
result.extend(results)
for i in result[:-2]:
    print(i,end=' ')
print(result[-1],end='')