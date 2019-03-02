import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class GuidedBackProp():
    def __init__(self, model, use_cuda):
        self.model = model.eval()
        self.use_cuda  = use_cuda
        if self.use_cuda:
            self.model = self.model.cuda()
        
        for module in self.model.named_modules():
            module[1].register_backward_hook(self.bp_relu)
        
    def bp_relu(self, module, grad_in, grad_out):
        if isinstance(module, nn.ReLU):
            return (torch.clamp(grad_in[0], min=0.0), )
    
    def __call__(self, x, index=None):
        x = x.clone()
        if self.use_cuda:
            x = x.cuda()
        x.requires_grad_()
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
            
        one_hot.backward()
        result = x.grad.cpu().numpy()[0]
        result = np.transpose(result, (1,2,0))
        return result, index

def arrange_img(img):
    img = np.maximum(img, 0)
    res = img - img.min()
    res /= res.max()
    res = np.uint8(res*255)
    return res