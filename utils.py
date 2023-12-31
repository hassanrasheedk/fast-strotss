import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from imageio import imread
import numpy as np
import os
import math
import PIL

use_random=True
# Tensor and PIL utils

def pil_loader(path):
    with open(path, 'rb') as f:
        img = PIL.Image.open(f)
        return img.convert('RGB')

def tensor_resample(tensor, dst_size, mode='bilinear'):
    return F.interpolate(tensor, dst_size, mode=mode, align_corners=False)

def pil_resize_short_edge_to(pil, trg_size):
    short_w = pil.width < pil.height
    ar_resized_short = (trg_size / pil.width) if short_w else (trg_size / pil.height)
    resized = pil.resize((int(pil.width * ar_resized_short), int(pil.height * ar_resized_short)), PIL.Image.BICUBIC)
    return resized

def pil_resize_long_edge_to(pil, trg_size):
    short_w = pil.width < pil.height
    ar_resized_long = (trg_size / pil.height) if short_w else (trg_size / pil.width)
    resized = pil.resize((int(pil.width * ar_resized_long), int(pil.height * ar_resized_long)), PIL.Image.BICUBIC)
    return resized

def np_to_pil(npy):
    return PIL.Image.fromarray(npy.astype(np.uint8))

def pil_to_np(pil):
    return np.array(pil)

def tensor_to_np(tensor, cut_dim_to_3=True):
    if len(tensor.shape) == 4:
        if cut_dim_to_3:
            tensor = tensor[0]
        else:
            return tensor.data.cpu().numpy().transpose((0, 2, 3, 1))
    return tensor.data.cpu().numpy().transpose((1,2,0))

def np_to_tensor(npy, space):
    if space == 'vgg':
        return np_to_tensor_correct(npy)
    return (torch.Tensor(npy.astype(np.float64) / 127.5) - 1.0).permute((2,0,1)).unsqueeze(0)

def np_to_tensor_correct(npy):
    pil = np_to_pil(npy)
    transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(pil).unsqueeze(0)

# Laplacian Pyramid

def laplacian(x):
    # x - upsample(downsample(x))
    return x - tensor_resample(tensor_resample(x, [x.shape[2] // 2, x.shape[3] // 2]), [x.shape[2], x.shape[3]])

def make_laplace_pyramid(x, levels):
    pyramid = []
    current = x
    for i in range(levels):
        pyramid.append(laplacian(current))
        current = tensor_resample(current, (max(current.shape[2] // 2,1), max(current.shape[3] // 2,1)))
    pyramid.append(current)
    return pyramid

def fold_laplace_pyramid(pyramid):
    current = pyramid[-1]
    for i in range(len(pyramid)-2, -1, -1): # iterate from len-2 to 0
        up_h, up_w = pyramid[i].shape[2], pyramid[i].shape[3]
        current = pyramid[i] + tensor_resample(current, (up_h,up_w))
    return current

def sample_indices(feat_content, feat_style_all, r, ri, xx, xy, yx):

    indices = None
    const = 128**2 # 32k or so

    feat_style =  feat_style_all[ri]

    for i in range(len(feat_style)):
        
        feat_cont = feat_content[i]
        d = feat_style[i].size(1)
        feat_style_st = feat_style[i].view(1,d,-1,1)
        big_size = feat_cont.shape[2] * feat_cont.shape[3] # num feaxels

        stride_x = int(max(math.floor(math.sqrt(big_size//const)),1))
        offset_x = np.random.randint(stride_x)
        stride_y = int(max(math.ceil(math.sqrt(big_size//const)),1))
        offset_y = np.random.randint(stride_y)
        xx_arr, xy_arr = np.meshgrid(np.arange(feat_cont.shape[2])[offset_x::stride_x], np.arange(feat_cont.shape[3])[offset_y::stride_y])

        xx_arr = np.expand_dims(xx_arr.flatten(),1)
        xy_arr = np.expand_dims(xy_arr.flatten(),1)
        xc = np.concatenate([xx_arr,xy_arr], 1)

        region_mask = r

        try:
            xc = xc[region_mask[xy_arr[:,0],xx_arr[:,0]], :]
        except:
            region_mask = region_mask[:,:]
            xc = xc[region_mask[xy_arr[:,0],xx_arr[:,0]], :]
        
        xx[ri].append(xc[:,0])
        xy[ri].append(xc[:,1])

        feat_result = np.arange(feat_style_st.size(2)).astype(np.int32)
        yx[ri].append(feat_result)

def get_feature_indices(xx_dict, xy_dict, yx_dict, ri=0, i=0, cnt=32**2):

    xx = xx_dict[ri][i][:cnt]
    xy = xy_dict[ri][i][:cnt]
    yx = yx_dict[ri][i][:cnt]

    return xx, xy, yx

def spatial_feature_extract(feat_result, feat_content, xx, xy):

    l2, l3 = [], []
    device = feat_result[0].device

    # for each extracted layer
    for i in range(len(feat_result)):
        fr = feat_result[i]
        fc = feat_content[i]

        # hack to detect reduced scale
        if i>0 and feat_result[i-1].size(2) > feat_result[i].size(2):
            xx = xx/2.0
            xy = xy/2.0


        # go back to ints and get residual
        xxm = np.floor(xx).astype(np.float32)
        xxr = xx - xxm

        xym = np.floor(xy).astype(np.float32)
        xyr = xy - xym

        # do bilinear resample
        w00 = torch.from_numpy((1.-xxr)*(1.-xyr)).float().view(1, 1, -1, 1).to(device)
        w01 = torch.from_numpy((1.-xxr)*xyr).float().view(1, 1, -1, 1).to(device)
        w10 = torch.from_numpy(xxr*(1.-xyr)).float().view(1, 1, -1, 1).to(device)
        w11 = torch.from_numpy(xxr*xyr).float().view(1, 1, -1, 1).to(device)

        xxm = np.clip(xxm.astype(np.int32),0,fr.size(2)-1)
        xym = np.clip(xym.astype(np.int32),0,fr.size(3)-1)

        s00 = xxm*fr.size(3)+xym
        s01 = xxm*fr.size(3)+np.clip(xym+1,0,fr.size(3)-1)
        s10 = np.clip(xxm+1,0,fr.size(2)-1)*fr.size(3)+(xym)
        s11 = np.clip(xxm+1,0,fr.size(2)-1)*fr.size(3)+np.clip(xym+1,0,fr.size(3)-1)

        fr = fr.view(1,fr.size(1),fr.size(2)*fr.size(3),1)
        fr = fr[:,:,s00,:].mul_(w00).add_(fr[:,:,s01,:].mul_(w01)).add_(fr[:,:,s10,:].mul_(w10)).add_(fr[:,:,s11,:].mul_(w11))

        fc = fc.view(1,fc.size(1),fc.size(2)*fc.size(3),1)
        fc = fc[:,:,s00,:].mul_(w00).add_(fc[:,:,s01,:].mul_(w01)).add_(fc[:,:,s10,:].mul_(w10)).add_(fc[:,:,s11,:].mul_(w11))

        l2.append(fr)
        l3.append(fc)

    x_st = torch.cat([li.contiguous() for li in l2],1)
    c_st = torch.cat([li.contiguous() for li in l3],1)

    xx = torch.from_numpy(xx).view(1,1,x_st.size(2),1).float().to(device)
    yy = torch.from_numpy(xy).view(1,1,x_st.size(2),1).float().to(device)
    
    x_st = torch.cat([x_st,xx,yy],1)
    c_st = torch.cat([c_st,xx,yy],1)
    return x_st, c_st

def rgb_to_yuv(rgb):
    C = torch.Tensor([[0.577350,0.577350,0.577350],[-0.577350,0.788675,-0.211325],[-0.577350,-0.211325,0.788675]]).to(rgb.device)
    yuv = torch.mm(C,rgb)
    return yuv

def pairwise_distances_cos(x, y):
    x_norm = torch.sqrt((x**2).sum(1).view(-1, 1))
    y_t = torch.transpose(y, 0, 1)
    y_norm = torch.sqrt((y**2).sum(1).view(1, -1))
    dist = 1.-torch.mm(x, y_t)/x_norm/y_norm
    return dist

def pairwise_distances_sq_l2(x, y):
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 1e-5, 1e5)/x.size(1)

def create_mask_from_image(image, ignore_color=[0, 0, 0]):
    """
    Create a mask from an image, where pixels matching the ignore_color are set to 0, and others to 1.

    :param image_path: Path to the input image.
    :param ignore_color: Color to be ignored, default is black ([0, 0, 0]).
    :return: Mask tensor of shape (1, H, W), where 1 indicates important areas and 0 indicates areas to ignore.
    """

    # Check if the image is grayscale or RGB
    if len(image.shape) == 2:  # Grayscale image
        mask = image != ignore_color[0]
    else:  # RGB image
        mask = np.all(image != ignore_color, axis=-1)

    mask = torch.from_numpy(mask).unsqueeze(0).float()

    return mask


def extract_regions(content_path, style_path):
    s_regions = imread(style_path).transpose(1,0,2)
    c_regions = imread(content_path).transpose(1,0,2)

    color_codes,c1 = np.unique(s_regions.reshape(-1, s_regions.shape[2]), axis=0,return_counts=True)

    color_codes = color_codes[c1>10000]

    c_out = []
    s_out = []

    for c in color_codes:
        c_expand =  np.expand_dims(np.expand_dims(c,0),0)
        
        s_mask = np.equal(np.sum(s_regions - c_expand,axis=2),0).astype(np.float32)
        c_mask = np.equal(np.sum(c_regions - c_expand,axis=2),0).astype(np.float32)

        s_out.append(s_mask)
        c_out.append(c_mask)

    return [c_out,s_out]