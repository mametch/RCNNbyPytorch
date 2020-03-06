import os
import sys
import numpy
import random
from PIL import Image
import torch
import torchvision
from torchvision.ops.boxes import nms
import torchvision.transforms.functional as F

th = 50
places_path = '/home/pcd001/data/places365_standard/train'
places_list = []
for (root, dirs, files) in os.walk(places_path):
    for file in files:
        places_list.append( os.path.join(root, file))
places_list = random.choices(places_list, k=50)

#dir_path = '/home/pcd001/data/FashionStyle14/val'
#seg_path = '/home/pcd001/data/FashionStyle14/val_seg'
#rem_path = '/home/pcd001/data/FashionStyle14/val_white'

dir_path = '/home/pcd001/data/ASFv2/ASF6/test'
seg_path = '/home/pcd001/data/ASFv2/ASF6/test_seg'
rem_path = '/home/pcd001/data/ASFv2/ASF6/test_places'
os.mkdir(rem_path)

dir_list = os.listdir(seg_path)
for dir in dir_list:
    rem_dir_path = os.path.join(rem_path, dir)
    os.mkdir(rem_dir_path)

    im_path = os.path.join(seg_path, dir)
    im_list = os.listdir(im_path)

    for im_name in im_list:
        seg_img = os.path.join(im_path, im_name)
        seg = Image.open(seg_img)
        mask = seg.convert("L")
        mask.point(lambda x: 0 if x < th else x)
        mask.point(lambda x: 255 if x >= th else x)

        dir_img = os.path.join(dir_path, dir, im_name)
        img = Image.open(dir_img)
        width, height = img.size

        for i, pl_path in enumerate(places_list):
            bg = Image.open(pl_path)
            bg = bg.resize((width, height))
            masked = Image.composite(img, bg, mask)

            save_path = os.path.join(rem_dir_path, im_name.replace('.jpg', '_') + str(i) + '.jpg')
            masked.save(save_path, quality=95)

        #bg = Image.new('RGB', (width, height), (255,255,255))
        #masked = Image.composite(img, bg, mask)
        #save_path = os.path.join(rem_dir_path, im_name)
        #masked.save(save_path, quality=95)

print("Finished!!")
