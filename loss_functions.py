import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import os
import math
import PIL
from utils import *

def distmat(x, y, cos_d=True):
    if cos_d:
        M = pairwise_distances_cos(x, y)
    else:
        M = torch.sqrt(pairwise_distances_sq_l2(x, y))
    return M

def content_loss(feat_result, feat_content):
    d = feat_result.size(1)

    X = feat_result.transpose(0,1).contiguous().view(d,-1).transpose(0,1)
    Y = feat_content.transpose(0,1).contiguous().view(d,-1).transpose(0,1)

    Y = Y[:,:-2]
    X = X[:,:-2]
    # X = X.t()
    # Y = Y.t()

    Mx = distmat(X, X)
    Mx = Mx#/Mx.sum(0, keepdim=True)

    My = distmat(Y, Y)
    My = My#/My.sum(0, keepdim=True)

    d = torch.abs(Mx-My).mean()# * X.shape[0]
    return d

def style_loss(X, Y, cos_d=True):
    d = X.shape[1]

    if d == 3:
        X = rgb_to_yuv(X.transpose(0,1).contiguous().view(d,-1)).transpose(0,1)
        Y = rgb_to_yuv(Y.transpose(0,1).contiguous().view(d,-1)).transpose(0,1)
    else:
        X = X.transpose(0,1).contiguous().view(d,-1).transpose(0,1)
        Y = Y.transpose(0,1).contiguous().view(d,-1).transpose(0,1)

    # Relaxed EMD
    CX_M = distmat(X, Y, cos_d=True)

    if d==3: CX_M = CX_M + distmat(X, Y, cos_d=False)

    m1, m1_inds = CX_M.min(1)
    m2, m2_inds = CX_M.min(0)

    remd = torch.max(m1.mean(), m2.mean())

    return remd

def moment_loss(X, Y, moments=[1,2]):
    loss = 0.
    X = X.squeeze().t()
    Y = Y.squeeze().t()

    mu_x = torch.mean(X, 0, keepdim=True)
    mu_y = torch.mean(Y, 0, keepdim=True)
    mu_d = torch.abs(mu_x - mu_y).mean()

    if 1 in moments:
        # print(mu_x.shape)
        loss = loss + mu_d

    if 2 in moments:
        X_c = X - mu_x
        Y_c = Y - mu_y
        X_cov = torch.mm(X_c.t(), X_c) / (X.shape[0] - 1)
        Y_cov = torch.mm(Y_c.t(), Y_c) / (Y.shape[0] - 1)

        # print(X_cov.shape)
        # exit(1)

        D_cov = torch.abs(X_cov - Y_cov).mean()
        loss = loss + D_cov

    return loss

def calculate_loss(feat_result, feat_content, feat_style, feat_guidance, xx_dict, xy_dict, content_weight, regions, moment_weight=1.0):
    # spatial feature extract
    num_locations = 1024
    xx, xy = get_feature_indices(xx_dict, xy_dict, cnt=num_locations)
    spatial_result, spatial_content = spatial_feature_extract(feat_result, feat_content, xx[0], xy[1])

    if feat_guidance.sum() > 0:
        gxx, gyy = get_guidance_indices()
        g_spatial_result, g_spatial_content = spatial_feature_extract(feat_result, feat_content, gxx, gyy)

    loss_content = content_loss(spatial_result, spatial_content)

    d = feat_style.shape[1]
    spatial_style = feat_style.view(1, d, -1, 1)

    feat_max = 3+2*64+128*2+256*3+512*2 # (sum of all extracted channels)

    loss_remd = style_loss(spatial_result[:, :feat_max, :, :], spatial_style[:, :feat_max, :, :])

    if feat_guidance.sum() > 0.:
        for j in range(feat_guidance.size(2)):
            loss_remd += style_loss(g_spatial_result[:,:-2,j:(j+1),:], feat_guidance[:,:,j:(j+1),:])[0]/feat_guidance.size(2)
    if feat_guidance.sum() > 0.:
        loss_moment = moment_loss(torch.cat([spatial_result, g_spatial_result],2)[:,:-2,:,:], torch.cat([spatial_style, feat_guidance],2), moments=[1,2]) # -2 is so that it can fit?
    else:
        loss_moment = moment_loss(spatial_result[:,:-2,:,:], spatial_style, moments=[1,2]) # -2 is so that it can fit?
    # palette matching
    content_weight_frac = 1./max(content_weight,1.)
    loss_moment += content_weight_frac * style_loss(spatial_result[:,:3,:,:], spatial_style[:,:3,:,:])
    
    loss_style = loss_remd + moment_weight * loss_moment
    # print(f'Style: {loss_style.item():.3f}, Content: {loss_content.item():.3f}')

    style_weight = 1.0 + moment_weight
    loss_total = (content_weight * loss_content + loss_style) / (content_weight + style_weight)
    return loss_total