import os
import sys
import glob
import numpy as np
import argparse
from PIL import Image, ImageDraw
import torch
import torchvision
from torchvision.ops.boxes import nms
import torchvision.transforms.functional as F

parser = argparse.ArgumentParser(description='Mask R-CNN with Pytorch and Torchvision')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dir_path', default='../images', type=str,
                    help='input image path')
parser.add_argument('--score_th', default=0.8, type=float,
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

# Get image absolute paths
img_list = glob.glob(os.path.join(args.dir_path, '*'))
for img_path in img_list:

    # load image and input
    name, ext = os.path.splitext(img_path)
    img = Image.open(img_path)
    image_tensor = torchvision.transforms.functional.to_tensor(img)
    with torch.no_grad():
        pred = model([image_tensor.to(device)])[0]

    # Excluding bboxes with low confidence scores
    ToF = np.where(pred['scores'].cpu() > args.score_th, True, False)
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

        # Draw bbox
        x0, y0, x1, y1 = boxes[ind].round()
        img_cp = img.copy()
        d = ImageDraw.Draw(img_cp)
        d.rectangle([(x0, y0), (x1, y1)], outline='green', width=3)

        # Save the image drawn bbox
        save_path = name + '_bbox' + str(i) + ext
        img_cp.save(save_path)

        # Convert the output mask image to PIL and save
        mask_img = F.to_pil_image(masks[ind].cpu())
        save_path = name + '_mask' + str(i) + ext
        mask_img.save(save_path)
