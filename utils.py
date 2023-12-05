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

def sample_indices(feat_content, feat_style_all, r, ri):

    indices = None
    const = 128**2 # 32k or so

    # feat_style =  feat_style_all[ri]

    # feat_dims = feat_style.shape[1]
    big_size = feat_content.shape[2] * feat_content.shape[3] # num feaxels

    stride_x = int(max(math.floor(math.sqrt(big_size//const)),1))
    offset_x = np.random.randint(stride_x)
    stride_y = int(max(math.ceil(math.sqrt(big_size//const)),1))
    offset_y = np.random.randint(stride_y)
    xx, xy = np.meshgrid(np.arange(feat_content.shape[2])[offset_x::stride_x], np.arange(feat_content.shape[3])[offset_y::stride_y])

    xx = np.expand_dims(xx.flatten(),1)
    xy = np.expand_dims(xy.flatten(),1)
    xc = np.concatenate([xx,xy], 1)

    region_mask = r

    # Debugging the shapes
    print(f"Shape of region_mask: {region_mask.shape}")
    print(f"Shape of xx: {xx.shape}")
    print(f"Shape of xy: {xy.shape}")
    print(f"Shape of xc: {xc.shape}")

    try:
        xc = xc[region_mask[xy[:,0],xx[:,0]], :]
    except:
        region_mask = region_mask[:,:]
        xc = xc[region_mask[xy[:,0],xx[:,0]], :]
    
    return xc[:,0], xc[:,1]

def get_feature_indices(xx_dict, xy_dict, ri=0, i=0, cnt=32**2):

        global use_random

        if use_random:
            xx = xx_dict[ri][i][:cnt]
            xy = xy_dict[ri][i][:cnt]
            # yx = self.rand_iy[ri][i][:cnt]
        else:
            xx = xx_dict[ri][i][::(xx_dict[ri][i].shape[0]//cnt)]
            xy = xy_dict[ri][i][::(xy_dict[ri][i].shape[0]//cnt)]
            # yx =  self.rand_iy[ri][i][::(self.rand_iy[ri][i].shape[0]//cnt)]

        return xx, xy

    
def get_guidance_indices(feat_result, coords):

    xx = (coords[:,0]*feat_result.size(2)).astype(np.int64)
    xy = (coords[:,1]*feat_result.size(3)).astype(np.int64)

    return xx, xy

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



def load_style_guidance(extractor,style_im,coords_t,device="cuda:0"):

    coords = coords_t.copy()
    coords[:,0]=coords[:,0]*style_im.size(2)
    coords[:,1]=coords[:,1]*style_im.size(3)
    coords = coords.astype(np.int64)

    xx = coords[:,0]
    xy = coords[:,1]

    zt = extractor(style_im)
    
    l2 = []

    for i in range(len(zt)):

        temp = zt[i]

        if i>0 and zt[i-1].size(2) > zt[i].size(2):
            xx = xx/2.0
            xy = xy/2.0

        xxm = np.floor(xx).astype(np.float32)
        xxr = xx - xxm

        xym = np.floor(xy).astype(np.float32)
        xyr = xy - xym

        w00 = torch.from_numpy((1.-xxr)*(1.-xyr)).float().view(1,1,-1,1).to(device)
        w01 = torch.from_numpy((1.-xxr)*xyr).float().view(1,1,-1,1).to(device)
        w10 = torch.from_numpy(xxr*(1.-xyr)).float().view(1,1,-1,1).to(device)
        w11 = torch.from_numpy(xxr*xyr).float().view(1,1,-1,1).to(device)


        xxm = np.clip(xxm.astype(np.int32),0,temp.size(2)-1)
        xym = np.clip(xym.astype(np.int32),0,temp.size(3)-1)

        s00 = xxm*temp.size(3)+xym
        s01 = xxm*temp.size(3)+np.clip(xym+1,0,temp.size(3)-1)
        s10 = np.clip(xxm+1,0,temp.size(2)-1)*temp.size(3)+(xym)
        s11 = np.clip(xxm+1,0,temp.size(2)-1)*temp.size(3)+np.clip(xym+1,0,temp.size(3)-1)


        temp = temp.view(1,temp.size(1),temp.size(2)*temp.size(3),1)
        temp = temp[:,:,s00,:].mul_(w00).add_(temp[:,:,s01,:].mul_(w01)).add_(temp[:,:,s10,:].mul_(w10)).add_(temp[:,:,s11,:].mul_(w11))
        
        l2.append(temp)
    gz = torch.cat([li.contiguous() for li in l2],1)

    return gz

def load_style_folder(extractor, style_im, regions, ri, n_samps=-1,subsamps=-1,scale=-1, inner=1, cpu_mode=False):
        
    total_sum = 0.
    z = []
    z_ims = []
    nloaded = 0

    nloaded += 1
    
    r_temp = regions[1][ri]
    if len(r_temp.shape) > 2:
        r_temp = r_temp[:,:,0]

    r_temp = torch.from_numpy(r_temp).unsqueeze(0).unsqueeze(0).contiguous()
    r = F.interpolate(r_temp,(style_im.size(3),style_im.size(2)),mode='bilinear', align_corners=False)[0,0,:,:].numpy()        
    sts = [style_im]

    z_ims.append(style_im)

    for j in range(inner):

        style_im = sts[np.random.randint(0,len(sts))]
        
        with torch.no_grad():
            zt = extractor.forward_cat(style_im,r,samps=subsamps)
            
        zt = [li.view(li.size(0),li.size(1),-1,1) for li in zt]

        # if len(z) == 0:
        #     z = zt

        # else:
        #     z = [torch.cat([zt[i],z[i]],2) for i in range(len(z))]

    return zt