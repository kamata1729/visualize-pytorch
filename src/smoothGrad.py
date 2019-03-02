import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F

class SmoothGrad():
    def __init__(self, model, use_cuda, stdev_spread=0.15, n_samples=25, magnitude=True):
        self.model = model.eval()
        self.use_cuda = use_cuda
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples
        self.magnitude = magnitude
        if self.use_cuda:
            self.model = self.model.cuda()
    
    def __call__(self, x, index=None):
        x = x.clone()
        if self.use_cuda:
            x = x.cuda()
        stdev = self.stdev_spread/(x.max() - x.min())
        std_tensor = torch.ones_like(x) * stdev
        total_gradients = torch.zeros_like(x)
        for i in tqdm(range(self.n_samples)):
            x_plus_noise = torch.normal(mean=x, std=std_tensor)
            x_plus_noise.requires_grad_()
            if self.use_cuda:
                x_plus_noise = x_plus_noise.cuda()
            
            output = self.model(x_plus_noise)
            
            if not index:
                index = np.argmax(output.cpu().data.numpy())
            
            one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
            one_hot[0][index] = 1
            one_hot = torch.from_numpy(one_hot)
            one_hot.requires_grad_()
            if self.use_cuda:
                one_hot = torch.sum(one_hot.cuda() * output)
            else:
                one_hot = torch.sum(one_hot * output)
                
            if x_plus_noise.grad is not None:
                x_plus_noise.grad.data.zero_()
            one_hot.backward()
            
            grad = x_plus_noise.grad
            
            if self.magnitude:
                total_gradients += (grad * grad)
            else:
                total_gradients += grad
                
        avg_gradients = total_gradients[0, :, :, :] / self.n_samples
        avg_gradients = np.transpose(avg_gradients.cpu().numpy(), (1,2,0))
        return avg_gradients, index

def show_as_gray_image(img,percentile=99):
    img_2d = np.sum(img, axis=2)
    span = abs(np.percentile(img_2d, percentile))
    vmin = -span
    vmax = span
    img_2d = np.clip((img_2d - vmin) / (vmax - vmin), -1, 1)
    return np.uint8(img_2d*255)
