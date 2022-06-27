# -*- coding: utf-8 -*-
"""
Created on Sat May 23 09:08:35 2020

@author: mouse
"""

from torchvision import transforms
import torch
from PIL import Image
import random
import torch.nn as nn
from torch.distributions import Beta
import numpy as np
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import tools.VGG as VGG
from tools.CBST_model import *
import mmcv

style_path = "/home/dailh/WCT2/water_quality"
# class waterModel(object):
#     def __init__(self,vgg_checkpoint,model_path):
#         self.JBL = Model().to('cuda')
#         self.JBL.load_state_dict(torch.load(model_path))
#         self.vgg = VGG.vgg
#         self.vgg.load_state_dict(torch.load(vgg_checkpoint))
#         vgg = nn.Sequential(*list(self.vgg.children())[:31])
#         self.net = VGG.Net(vgg).to('cuda')
#         self.style_imgs = self.style_init()
#
#     def style_init(self):
#         transform = transforms.Compose([
#             transforms.Resize((512, 512), Image.BICUBIC),
#             transforms.ToTensor()
#         ])
#         style_imgs = []
#         for style_id in range(1,8):
#             style_img_path = os.path.join(style_path, 'type' + str(style_id) + '.jpg')
#             style_imgs.append(transform(Image.open(style_img_path).convert('RGB')).cuda())
#         return style_imgs
#
#     def forward(self,cont_img,style_id):
#         style_img = self.style_imgs[style_id-1]
#         cont_img = resize(cont_img.squeeze(),512)
#         low_cont = resize(cont_img, cont_img.shape[-1] // 2).unsqueeze(0)
#         low_style = resize(style_img, style_img.shape[-1] // 2).unsqueeze(0)
#
#         self.JBL.eval()
#         cont_feat = self.net.encode_with_intermediate(low_cont)
#         style_feat = self.net.encode_with_intermediate(low_style)
#
#         _, output = self.JBL(cont_img.unsqueeze(0), cont_feat, style_feat)
#         return output

class CBST_Model(object):
    def __init__(self,vgg_checkpoint,model_path,style_num):
        self.JBL = Model(style_num).to('cuda')
        self.JBL.load_state_dict(torch.load(model_path))
        self.vgg = VGG.vgg
        self.vgg.load_state_dict(torch.load(vgg_checkpoint))
        vgg = nn.Sequential(*list(self.vgg.children())[:31])
        self.net = VGG.Net(vgg).to('cuda').eval()

    def forward(self,cont_img,style_id):
        cont_img_ori = cont_img.clone()
        cont_img = resize(cont_img.squeeze(),512)
        low_cont = resize(cont_img, cont_img.shape[-1] // 2).unsqueeze(0)

        self.JBL.eval()
        cont_feat = self.net.encode_with_intermediate(low_cont)

        # _, output = self.JBL(cont_img.unsqueeze(0), cont_feat, [style_id])
        _, output = self.JBL(cont_img_ori, cont_feat, [style_id])
        return output



class WaterTransfer(object):
    def __init__(self,model_path,style_num):
        self.model_path = model_path
        self.style_num = style_num
        # self.model = TransformerNet(style_num = self.style_num)
        self.model = CBST_Model("/home/dailh/Joint-Bilateral-Learning/checkpoints/vgg_normalised.pth",self.model_path,style_num)
        # self.model = waterModel("/home/dailh/Joint-Bilateral-Learning/checkpoints/vgg_normalised.pth","/home/dailh/Joint-Bilateral-Learning/checkpoints/ckpt_maskloss_8.pth")


        self.mean = torch.tensor([123.675, 116.28, 103.53]).cuda()
        self.std = torch.tensor([58.395, 57.12, 57.375]).cuda()
        self.distortion  = PhotoMetricDistortion()
        return

    def denormalize(self, img):
        img = img * self.std.view(3, 1, 1).expand(img.shape)
        img = img + self.mean.view(3, 1, 1).expand(img.shape)
        img /= 255.0
        return img

    def normalize(self, img):
        img *= 255.0
        img = img - self.mean.view(3, 1, 1).expand(img.shape)
        img = img / self.std.view(3, 1, 1).expand(img.shape)

        return img


    def resize(self,image, size):
        image = F.interpolate(image.unsqueeze(0), size=size, mode="bilinear").squeeze(0)
        return image

    def forward(self,img, style_id = None):
        # if random.random() < 0.125:
        #     return resize(img.squeeze(),512),0
        if style_id == None:
            style_id = random.randint(0,self.style_num-1)
        # content_transform = transforms.Compose([
        #     # transforms.ToTensor(),
        #     transforms.Lambda(lambda x: x.mul(255))
        # ])
        # img = content_transform(img)

        # domain_mix
        img = img.unsqueeze(0)
        with torch.no_grad():
            # img = self.model(img, style_id=[style_id]).squeeze()
            img = self.model.forward(img, style_id).squeeze()
        # img /= 255.0
        # save_image(img,'out3.jpg')
        return img, style_id

    def domain_mix(self, img, mix_img, style_id_1):
        m = Beta(torch.FloatTensor([2]), torch.FloatTensor([2]))
        style_id_2 = style_id_1
        while style_id_2 == style_id_1:
            style_id_2 = random.randint(0,self.style_num-1)
        weight = m.sample(sample_shape=torch.Size([1])).cuda()
        img, _ = self.forward(img, style_id_2)
        if np.random.random() < 1:
            return img
        if weight > 0.5:
            weight = 1 - weight
        img = mix_img * weight + img * (1 - weight)
        return img

    def channel_swap_mix(self,img):
        if random.random() < 0.5:
            return img
        img = img.cpu()
        arr = np.arange(3)
        # np.random.shuffle(arr)
        arr = np.array([0,2,1])
        # swap and mixup
        if random.random() < 0:
            m = Beta(torch.FloatTensor([2]), torch.FloatTensor([2]))
            weight = m.sample(sample_shape=torch.Size([1]))
            term = img.clone()
            img[arr] = img
            if weight > 0.5:
                weight = 1 - weight
            img = img * weight + term * (1 - weight)
        else:
            # just swap
            img[arr] = img
        img = img.cuda()
        return img

    def loop_forward(self, imgs, style_ids = None, img_size = 512):
        imgs_list = []
        for i in range(imgs.shape[0]):
            if style_ids == None:
                style_id = 10
            else:
                style_id = int(style_ids[i])
            img = imgs[i]
            img = self.denormalize(img)
            # save_image(img, 'input.jpg')

            # domain mix
            if random.random() < 0.8:
                # img = self.domain_mix(img, img, style_id)
                img,_ = self.forward(img)
            # elif random.random() < 0.5:
            #     img = self.distortion(img)
            # save_image(img, 'peep1.jpg')
            # img = self.channel_swap_mix(img)
            img = self.normalize(img)
            # img = self.resize(img, img_size)


            imgs_list.append(img)
        imgs = torch.stack(imgs_list)
        return imgs

class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18
                 ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, img):
        img *= 255
        # tensor 2 bgr numpy
        order = np.array([2, 1, 0])
        img = img[order]
        img = img.permute(1,2,0).detach().cpu().numpy()
        # mmcv.imwrite(img,'in.jpg')

        # random brightness
        if np.random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(2)
        if mode == 1:
            if np.random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if np.random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if np.random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if np.random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if np.random.randint(2):
            img = img[..., np.random.permutation(3)]

        # BGR 2 tensor
        # mmcv.imwrite(img,'out.jpg')

        img = torch.tensor(img).cuda()
        img = img.permute(2, 0, 1)
        order = np.array([2, 1, 0])
        img = img[order]
        img /= 255.
        return img