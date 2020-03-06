import os
import sys
import numpy
from PIL import Image, ImageDraw
import torch
import torchvision
from torchvision.ops.boxes import nms
#import torchvision.transforms.functional as F

nms_th = 0.5
score_th = 0.8
img_path = 'images/sample.jpg'

# chose device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# pretrained COCO
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

# load image and input
img = Image.open(img_path)
image_tensor = torchvision.transforms.functional.to_tensor(img)
with torch.no_grad():
    pred = model([image_tensor.to(device)])

# Retrieve bounding box and confidence score
boxes = pred[0]['boxes']
scores = pred[0]['scores']

# Non-Maximum Suppression (Reduce bounding box)
index = nms(boxes, scores, nms_th)

# Process one bbox at a time
d = ImageDraw.Draw(img)
for ind in index:
    # Excluding bboxes with low confidence scores
    if scores[ind] < score_th: continue
    
    # Excluding non-person labels (Person label is "1")
    if pred[0]['labels'][ind] != 1: continue

    # Draw bbox
    x0, y0, x1, y1 = boxes[ind].round()
    d.rectangle([(x0, y0), (x1, y1)], outline='green', width=3)

# Save the image drawn bbox
name, ext = os.path.splitext(img_path)[0]
save_path = name + '_bbox' + ext
img.save(save_path)

# Convert the output mask image to PIL and save
mask_img = F.to_pil_image(pred[0]['masks'][ind].cpu())
save_path = name + '_mask' + ext
mask_img.save(save_path)
