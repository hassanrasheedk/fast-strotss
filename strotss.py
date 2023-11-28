import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
from time import time
from argparse import ArgumentParser
from utils import *
from loss_functions import *

class Vgg16_Extractor(nn.Module):
    def __init__(self, space):
        super().__init__()
        self.vgg_layers = models.vgg16(pretrained=True).features

        for param in self.parameters():
            param.requires_grad = False
        self.capture_layers = [1,3,6,8,11,13,15,22,29]
        self.space = space
        
    def forward_base(self, x):
        feat = [x]
        for i in range(len(self.vgg_layers)):
            x = self.vgg_layers[i](x)
            if i in self.capture_layers: feat.append(x)
        return feat

    def forward(self, x):
        if self.space != 'vgg':
            x = (x + 1.) / 2.
            x = x - (torch.Tensor([0.485, 0.456, 0.406]).to(x.device).view(1, -1, 1, 1))
            x = x / (torch.Tensor([0.229, 0.224, 0.225]).to(x.device).view(1, -1, 1, 1))
        feat = self.forward_base(x)
        return feat
    
    def forward_samples_hypercolumn(self, X, samps=100):
        feat = self.forward(X)

        xx,xy = np.meshgrid(np.arange(X.shape[2]), np.arange(X.shape[3]))
        xx = np.expand_dims(xx.flatten(),1)
        xy = np.expand_dims(xy.flatten(),1)
        xc = np.concatenate([xx,xy],1)
        
        samples = min(samps,xc.shape[0])

        np.random.shuffle(xc)
        xx = xc[:samples,0]
        yy = xc[:samples,1]

        feat_samples = []
        for i in range(len(feat)):

            layer_feat = feat[i]

            # hack to detect lower resolution
            if i>0 and feat[i].size(2) < feat[i-1].size(2):
                xx = xx/2.0
                yy = yy/2.0

            xx = np.clip(xx, 0, layer_feat.shape[2]-1).astype(np.int32)
            yy = np.clip(yy, 0, layer_feat.shape[3]-1).astype(np.int32)

            features = layer_feat[:,:, xx[range(samples)], yy[range(samples)]]
            feat_samples.append(features.clone().detach())

        feat = torch.cat(feat_samples,1)
        return feat
    
def optimize(result, content, style, scale, content_weight, lr, extractor):
    # torch.autograd.set_detect_anomaly(True)
    result_pyramid = make_laplace_pyramid(result, 5)
    result_pyramid = [l.data.requires_grad_() for l in result_pyramid]

    opt_iter = 200
    # if scale == 1:
    #     opt_iter = 800

    # use rmsprop
    optimizer = optim.RMSprop(result_pyramid, lr=lr)

    # extract features for content
    feat_content = extractor(content) # 

    stylized = fold_laplace_pyramid(result_pyramid)
    # let's ignore the regions for now
    # some inner loop that extracts samples
    feat_style = None
    for i in range(5):
        with torch.no_grad():
            # r is region of interest (mask)
            feat_e = extractor.forward_samples_hypercolumn(style, samps=1000)
            feat_style = feat_e if feat_style is None else torch.cat((feat_style, feat_e), dim=2)
    # feat_style.requires_grad_(False)

    # init indices to optimize over
    xx, xy = sample_indices(feat_content[0], feat_style) # 0 to sample over first layer extracted
    for it in range(opt_iter):
        optimizer.zero_grad()

        stylized = fold_laplace_pyramid(result_pyramid)
        # original code has resample here, seems pointless with uniform shuffle
        # ...
        # also shuffle them every y iter
        if it % 1 == 0 and it != 0:
            np.random.shuffle(xx)
            np.random.shuffle(xy)
        feat_result = extractor(stylized)

        loss = calculate_loss(feat_result, feat_content, feat_style, [xx, xy], content_weight)
        loss.backward()
        optimizer.step()
    return stylized


def strotss(content_pil, style_pil, content_weight=1.0*16.0, device='cuda:0', space='uniform'):
    content_np = pil_to_np(content_pil)
    style_np = pil_to_np(style_pil)
    content_full = np_to_tensor(content_np, space).to(device)
    style_full = np_to_tensor(style_np, space).to(device)

    lr = 2e-3
    extractor = Vgg16_Extractor(space=space).to(device)

    scale_last = max(content_full.shape[2], content_full.shape[3])
    scales = []
    for scale in range(10):
        divisor = 2**scale
        if min(content_pil.width, content_pil.height) // divisor >= 33:
            scales.insert(0, divisor)
    
    for scale in scales:
        # rescale content to current scale
        content = tensor_resample(content_full, [ content_full.shape[2] // scale, content_full.shape[3] // scale ])
        style = tensor_resample(style_full, [ style_full.shape[2] // scale, style_full.shape[3] // scale ])
        print(f'Optimizing at resoluton [{content.shape[2]}, {content.shape[3]}]')

        # upsample or initialize the result
        if scale == scales[0]:
            # first
            result = laplacian(content) + style.mean(2,keepdim=True).mean(3,keepdim=True)
        elif scale == scales[-1]:
            # last 
            result = tensor_resample(result, [content.shape[2], content.shape[3]])
            lr = 1e-3
        else:
            result = tensor_resample(result, [content.shape[2], content.shape[3]]) + laplacian(content)

        # do the optimization on this scale
        result = optimize(result, content, style, scale, content_weight=content_weight, lr=lr, extractor=extractor)

        # next scale lower weight
        content_weight /= 2.0

    clow = -1.0 if space == 'uniform' else -1.7
    chigh = 1.0 if space == 'uniform' else 1.7
    result_image = tensor_to_np(tensor_resample(torch.clamp(result, clow, chigh), [content_full.shape[2], content_full.shape[3]])) # 
    # renormalize image
    result_image -= result_image.min()
    result_image /= result_image.max()
    return np_to_pil(result_image * 255.)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("content", type=str)
    parser.add_argument("style", type=str)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="strotss.png")
    parser.add_argument("--device", type=str, default="cuda:0")
    # uniform ospace = optimization done in [-1, 1], else imagenet normalized space
    parser.add_argument("--ospace", type=str, default="uniform", choices=["uniform", "vgg"])
    parser.add_argument("--resize_to", type=int, default=512)
    args = parser.parse_args()

    # make 256 the smallest possible long side, will still fail if short side is <
    if args.resize_to < 2**8:
        print("Resulution too low.")
        exit(1)

    content_pil, style_pil = pil_loader(args.content), pil_loader(args.style)
    content_weight = args.weight * 16.0

    device = args.device

    start = time()
    result = strotss(pil_resize_long_edge_to(content_pil, args.resize_to), 
                     pil_resize_long_edge_to(style_pil, args.resize_to), content_weight, device, args.ospace)
    result.save(args.output)
    print(f'Done in {time()-start:.3f}s')
