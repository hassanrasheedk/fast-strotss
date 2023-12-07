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
    
    def forward_cat(self, X, r, samps=100, forward_func=None):

        if not forward_func:
            forward_func = self.forward

        x = X
        out2 = forward_func(X)

        try:
            r = r[:,:,0]
        except:
            pass

        if r.max()<0.1:
            region_mask = np.greater(r.flatten()+1.,0.5)
        else:
            region_mask = np.greater(r.flatten(),0.5)

        xx,xy = np.meshgrid(np.array(range(x.size(2))), np.array(range(x.size(3))) )
        xx = np.expand_dims(xx.flatten(),1)
        xy = np.expand_dims(xy.flatten(),1)
        xc = np.concatenate([xx,xy],1)
        
        xc = xc[region_mask,:]

        const2 = min(samps,xc.shape[0])


        global use_random
        if use_random:
            np.random.shuffle(xc)
        else:
            xc = xc[::(xc.shape[0]//const2),:]

        xx = xc[:const2,0]
        yy = xc[:const2,1]

        temp = X
        temp_list = [ temp[:,:, xx[j], yy[j]].unsqueeze(2).unsqueeze(3) for j in range(const2)]
        temp = torch.cat(temp_list,2)

        l2 = []
        for i in range(len(out2)):

            temp = out2[i]

            if i>0 and out2[i].size(2) < out2[i-1].size(2):
                xx = xx/2.0
                yy = yy/2.0

            xx = np.clip(xx,0,temp.size(2)-1).astype(np.int32)
            yy = np.clip(yy,0,temp.size(3)-1).astype(np.int32)

            temp_list = [ temp[:,:, xx[j], yy[j]].unsqueeze(2).unsqueeze(3) for j in range(const2)]
            temp = torch.cat(temp_list,2)

            l2.append(temp.clone().detach())

        out2 = [torch.cat([li.contiguous() for li in l2],1)]

        return out2
    
def optimize(result, content, style, content_path, style_path, scale, content_weight, lr, extractor, coords=0, use_guidance=False, regions=0):
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
    ### Extract guidance features if required ###
    feat_guidance = np.array([0.])
    if use_guidance:
        feat_guidance = load_style_guidance(extractor, style_path, coords[:,2:], scale)
    
    # some inner loop that extracts samples
    feat_style = None
    for ri in range(len(regions[1])):
        with torch.no_grad():
            feat_e = load_style_folder(extractor, style, regions, ri, n_samps=1, subsamps=1000, inner=5)        
            feat_style = feat_e if feat_style is None else torch.cat((feat_style, feat_e), dim=2)

    if feat_style:
        feat_style = torch.cat(feat_style, dim=2)
    else:
        feat_style = torch.tensor([])

    for ri in range(len(regions[0])):
        r_temp = regions[0][ri]
        r_temp = torch.from_numpy(r_temp).unsqueeze(0).unsqueeze(0).contiguous()
        r = tensor_resample(r_temp, ([stylized.size(3), stylized.size(2)]))[0,0,:,:].numpy()     

        if r.max()<0.1:
            r = np.greater(r+1.,0.5)
        else:
            r = np.greater(r,0.5)

        xx = {}
        xy = {}

        xx_arr, xy_arr = sample_indices(feat_content, feat_style, r, ri) # 0 to sample over first layer extracted
        
        try:
            temp = xx[ri]
        except:
            xx[ri] = []
            xy[ri] = []

        xx[ri].append(xx_arr)
        xy[ri].append(xy_arr)

    # init indices to optimize over
    # xx, xy = sample_indices(feat_content[0], feat_style) # 0 to sample over first layer extracted
    for it in range(opt_iter):
        optimizer.zero_grad()
        stylized = fold_laplace_pyramid(result_pyramid)
        # original code has resample here, seems pointless with uniform shuffle
        # ...
        # also shuffle them every y iter
        if it % 1 == 0 and it != 0:
            for ri in xx.keys():
                np.random.shuffle(xx[ri])
                np.random.shuffle(xy[ri])

        feat_result = extractor(stylized)

        loss = calculate_loss(feat_result, feat_content, feat_style, feat_guidance, xx, xy, content_weight, regions)
        loss.backward()
        optimizer.step()
    return stylized


def strotss(content_pil, style_pil, content_path, style_path, regions, coords, content_weight=1.0*16.0, device='cuda:0', space='uniform', use_guidance=False, content_mask=None, style_mask=None):
    content_np = pil_to_np(content_pil)
    style_np = pil_to_np(style_pil)
    content_full = np_to_tensor(content_np, space).to(device)
    style_full = np_to_tensor(style_np, space).to(device)

    if content_mask is not None and style_mask is not None:
        content_mask = content_mask.to(device)
        style_mask = style_mask.to(device)

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
        result = optimize(result, content, style, content_path, style_path, scale, content_weight=content_weight, lr=lr, extractor=extractor, coords=coords, use_guidance=use_guidance, regions=regions)

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
    parser.add_argument("--content_mask", type=str, default=None)
    parser.add_argument("--style_mask", type=str, default=None)
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

    use_guidance = False
    coords=0.
    content_pil, style_pil = pil_loader(args.content), pil_loader(args.style)
    content_mask, style_mask = None, None

    if args.content_mask and args.style_mask is not None:
        use_guidance = True
        regions = extract_regions(args.content_mask, args.style_mask)

        pil_content_mask = pil_loader(args.content_mask)
        pil_style_mask = pil_loader(args.style_mask)

        pil_content_mask = pil_resize_long_edge_to(pil_content_mask, args.resize_to)
        pil_style_mask = pil_resize_long_edge_to(pil_style_mask, args.resize_to)

        content_mask = pil_to_np(pil_content_mask)
        style_mask = pil_to_np(pil_style_mask)

        content_mask = create_mask_from_image(content_mask)
        style_mask = create_mask_from_image(style_mask)
    else:
        try:
            regions = [[pil_to_np(pil_resize_long_edge_to(pil_loader(args.content), args.resize_to))[:,:,0]*0.+1.], [pil_to_np(pil_resize_long_edge_to(pil_loader(args.style), args.resize_to))[:,:,0]*0.+1.]]
        except:
            regions = [[pil_to_np(pil_resize_long_edge_to(pil_loader(args.content), args.resize_to))[:,:]*0.+1.], [pil_to_np(pil_resize_long_edge_to(pil_loader(args.style), args.resize_to))[:,:]*0.+1.]]
    
    content_weight = args.weight * 16.0

    device = args.device

    start = time()
    result = strotss(pil_resize_long_edge_to(content_pil, args.resize_to), 
                     pil_resize_long_edge_to(style_pil, args.resize_to), args.content, args.style, regions, coords, content_weight, device, args.ospace, use_guidance=use_guidance, content_mask=content_mask, style_mask=style_mask)
    result.save(args.output)
    print(f'Done in {time()-start:.3f}s')
