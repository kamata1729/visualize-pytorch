import cv2
import json
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models, transforms, utils

from src.gradCAM import *
from src.guidedBackProp import *
from src.smoothGrad import *

def run(image_path, index, cuda):
    raw_image = cv2.imread(image_path)[..., ::-1]
    raw_image = cv2.resize(raw_image, (224, 224))
    image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])(raw_image).unsqueeze(0)

    # gradCAM
    print("gradCAM...")
    grad_cam = GradCAM(models.resnet50(pretrained=True), 'layer4.2', use_cuda=cuda)
    cam, target_index = grad_cam(image, index=index)
    cam_on_image = show_cam_on_image(raw_image/255, cam)
    cv2.imwrite("results/gradCAM_" + image_path.split('/')[-1], cam_on_image)

    # guidedBackProp
    print("guidedBackProp...")
    guided_bp = GuidedBackProp(models.resnet50(pretrained=True), use_cuda=cuda)
    guided_cam, _ = guided_bp(image)
    cv2.imwrite("results/guidedbackProp_" + image_path.split('/')
                [-1], arrange_img(guided_cam))

    # guidedGradCAM
    print("guidedGradCAM...")
    guided_grad_cam = np.multiply(cam[..., None], guided_cam)
    cv2.imwrite("results/guidedGradCAM_" + image_path.split('/')
                [-1], arrange_img(guided_grad_cam))

    # smoothGrad
    print("smoothGrad...")
    smooth_grad = SmoothGrad(models.resnet50(
        pretrained=True), use_cuda=cuda, stdev_spread=0.2, n_samples=20)
    smooth_cam, _ = smooth_grad(image)
    cv2.imwrite("results/smoothGrad_" + image_path.split('/')
                [-1], show_as_gray_image(smooth_cam))
    
    # show index
    target_class = json.load(open("imagenet_class_index.json"))[str(target_index)]
    print("target_index: {}".format(target_index))
    print("target_class: {}".format(target_class))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='running gradCAM, guidedBP, smoothGrad')
    parser.add_argument('image_path', help='image path')
    parser.add_argument('--cuda', action='store_true', help='add this option to use gpu')
    parser.add_argument('--index', type=int, help='target imagenet index of gradCAM (see imagenet_class_index.json)')
    args = parser.parse_args()
    run(args.image_path, args.index, args.cuda)

    
