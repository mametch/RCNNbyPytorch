import os
import sys
import numpy as np
import argparse
from PIL import Image, ImageDraw
import torch
import torchvision
from torchvision.ops.boxes import nms
import torchvision.transforms.functional as F

parser = argparse.ArgumentParser(description='Keypoint R-CNN with Pytorch and Torchvision')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--img_path', default='../images/sample.jpg', type=str,
                    help='input image path')
parser.add_argument('--score_th', default=0.8, type=float,
                    help='Confidence score threshold')
parser.add_argument('--nms_th', default=0.5, type=float,
                    help='NMS threshold')
args = parser.parse_args()

# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# pretrained COCO
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

# load image and input
name, ext = os.path.splitext(args.img_path)
img = Image.open(args.img_path)
image_tensor = torchvision.transforms.functional.to_tensor(img)
with torch.no_grad():
    pred = model([image_tensor.to(device)])[0]

# Excluding bboxes with low confidence scores
ToF = np.where(pred['scores'].cpu() > args.score_th, True, False)
boxes = pred['boxes'][ToF]
scores = pred['scores'][ToF]
labels = pred['labels'][ToF]
keypoints = pred['keypoints']

# Non-Maximum Suppression (Reduce bounding box)
index = nms(boxes, scores, args.nms_th)

# Process one bbox at a time
for i, ind in enumerate(index):

    # Skip non-person labels (Person label is "1")
    if labels[ind] != 1: continue

    # Draw bbox
    x0, y0, x1, y1 = boxes[ind].round()
    img_cp = img.copy()
    d = ImageDraw.Draw(img_cp)
    d.rectangle([(x0, y0), (x1, y1)], outline='green', width=3)

    # Save the image drawn bbox
    save_path = name + '_bbox' + str(i) + ext
    img_cp.save(save_path)

    # Draw keypoints
    kps = keypoints[ind].cpu()
    img_cp = img.copy()
    d = ImageDraw.Draw(img_cp)
    for kp in kps:
        d.ellipse((kp[0]-3, kp[1]-3, kp[0]+3, kp[1]+3), fill='green')

    save_path = name + '_keypoint' + str(i) + ext
    img_cp.save(save_path)
