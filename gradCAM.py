import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class GradCAM():
    def __init__(self, model, target_layer, use_cuda):
        self.model = model.eval()
        self.target_layer = target_layer
        self.use_cuda = use_cuda
        self.feature_map = 0
        self.grad = 0
        
        if self.use_cuda:
            self.model = self.model.cuda()
        
        for module in self.model.named_modules():
            if module[0] == target_layer:
                module[1].register_forward_hook(self.save_feature_map)
                module[1].register_backward_hook(self.save_grad)
    
    def save_feature_map(self, module, input, output):
        self.feature_map =  output.detach()
        
    def save_grad(self, module, grad_in, grad_out):
        self.grad = grad_out[0].detach()
        
    def __call__(self, x, index=None):
        x = x.clone()
        if self.use_cuda:
            x = x.cuda()
            
        output = self.model(x)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
            
        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot)
        one_hot.requires_grad_()
        if self.use_cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)
            
        self.model.zero_grad()
        
        one_hot.backward()
        
        self.feature_map = self.feature_map.cpu().numpy()[0]
        
        self.weights = np.mean(self.grad.cpu().numpy(), axis = (2, 3))[0, :]
        
        cam = np.sum(self.feature_map * self.weights[:, None, None], axis=0)
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (x.size()[-1], x.size()[-2]))
        return cam, index

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
