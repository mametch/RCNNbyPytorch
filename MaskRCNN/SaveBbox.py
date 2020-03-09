import os
import sys
import glob
import json
import numpy as np
import argparse
from PIL import Image, ImageDraw
import torch
import torchvision
from torchvision.ops.boxes import nms
import torchvision.transforms.functional as F

parser = argparse.ArgumentParser(description='Mask R-CNN with Pytorch and Torchvision')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dir_path', default='images', type=str,
                    help='input image path')
parser.add_argument('--bbox_path', default='BboxOut', type=str,
                    help='input image path')
parser.add_argument('--mask_path', default='MaskOut', type=str,
                    help='input image path')
parser.add_argument('--score_th', default=0.5, type=float,
                    help='Confidence score threshold')
parser.add_argument('--nms_th', default=0.5, type=float,
                    help='NMS threshold')
args = parser.parse_args()

# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# pretrained COCO
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

# Saves
glob = glob.glob(os.path.join(args.dir_path, '**'), recursive=True)
dir_list = [p for p in glob if os.path.isdir(p)]
for p in dir_list:
    os.makedirs(p.replace(args.dir_path, args.bbox_path), exist_ok=True)
    os.makedirs(p.replace(args.dir_path, args.mask_path), exist_ok=True)

img_list = [p for p in glob if os.path.isfile(p)]
for img_path in img_list:
    # File existence check
    name, ext = os.path.splitext(img_path)
    if os.path.isfile(name.replace(args.dir_path, args.bbox_path) + '.json'):
        continue

    ImDict = {}

    # Load image and input
    img = Image.open(img_path)
    image_tensor = torchvision.transforms.functional.to_tensor(img)
    with torch.no_grad():
        pred = model([image_tensor.to(device)])[0]

    # Excluding bboxes with low confidence scores
    ToF = np.where(pred['scores'] > args.score_th, True, False)
    boxes = pred['boxes'][ToF]
    scores = pred['scores'][ToF]
    labels = pred['labels'][ToF]
    masks = pred['masks'][ToF]

    # Non-Maximum Suppression (Reduce bounding box)
    index = nms(boxes, scores, args.nms_th)

    # Process one bbox at a time
    for i, ind in enumerate(index):

        # Skip non-person labels (Person label is "1")
        if labels[ind] != 1: continue

        # Save bbox
        x0, y0, x1, y1 = boxes[ind].tolist()
        ImDict[i] = [scores[ind].tolist(), x0, y0, x1, y1]

        # Convert the output mask image to PIL and save
        mask_img = F.to_pil_image(masks[ind].cpu())
        save_path = name.replace(args.dir_path, args.mask_path) + '_' + str(i) + ext
        mask_img.save(save_path)

    # Save bbox
    with open(name.replace(args.dir_path, args.bbox_path) + '.json', 'w') as f:
        json.dump(ImDict, f)
