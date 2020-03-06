import os
import sys
import numpy
import random
from PIL import Image
import torch
import torchvision
from torchvision.ops.boxes import nms
import torchvision.transforms.functional as F

save_num = 1000000
nms_th = 0.5
w_th = 100
h_th = 100
score_th = 0.8

wo_path = '/home/pcd0001/data/FCDB_seg/original'
ws_path = '/home/pcd0001/data/FCDB_seg/segmentation'
os.mkdir(wo_path)
os.mkdir(ws_path)
im_path = '/home/pcd0001/data/FCDB_voc/JPEGImages'
#im_path = '/home/pcd0001/data/DeepFashion2/test/image'
im_list = os.listdir(im_path)
random.shuffle(im_list)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# pretrained COCO
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

i = 0
for im_name in im_list:
    path = os.path.join(im_path, im_name)
    im = Image.open(path)
    try:
        image_tensor = torchvision.transforms.functional.to_tensor(im)
    except:
        continue
    pred = model([image_tensor.to(device)])

    boxes = pred[0]['boxes']
    scores = pred[0]['scores']
    index = nms(boxes, scores, nms_th)

    j = 0
    for ind in index:
        if scores[ind] < score_th: continue
        if pred[0]['labels'][ind] != 1: continue

        x0, y0, x1, y1 = boxes[ind].round()
        if (x1 - x0) < w_th or (y1 - y0) < h_th: continue
        save_path = os.path.join(wo_path, im_name[:-4] + '_' + str(j) + '.jpg')
        im.crop((int(x0), int(y0), int(x1), int(y1))).save(save_path, quality=95)

        mask_im = F.to_pil_image(pred[0]['masks'][ind].cpu())
        mask_path = os.path.join(ws_path, im_name[:-4] + '_' + str(j) + '.jpg')
        mask_im.crop((int(x0), int(y0), int(x1), int(y1))).save(mask_path, quality=95)

        j += 1
        if (i + j) == save_num: break

    i += j
    if i == save_num: break
    if i % 1000 == 0: print("saved :", i)

print("Finished!!")
print(" save num :", i)
