from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np

from vgg import *
import math
import random
import pdb
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from dy_conv import Dynamic_conv2d
import torch.nn as nn
# from self_attention_cv import ViT, ResNet50ViT
# from self_attention_cv.transunet import TransUnet

from attention import SpatialAttention, ChannelwiseAttention

vgg_conv1_2 = vgg_conv2_2 = vgg_conv3_3 = vgg_conv4_3 = vgg_conv5_3 = None

import torch
import torch.nn as nn


EPSILON = 1e-10


def var(x, dim=0):
    x_zero_meaned = x - x.mean(dim).expand_as(x)
    return x_zero_meaned.pow(2).mean(dim)


class MultConst(nn.Module):
    def forward(self, input):
        return 255*input

class Fusion_ADD(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        temp = en_ir + en_vi
        return temp


class Fusion_AVG(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        temp = (en_ir + en_vi) / 2
        return temp


class Fusion_MAX(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        temp = torch.max(en_ir, en_vi)
        return temp


class Fusion_SPA(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        shape = en_ir.size()
        spatial_type = 'mean'
        # calculate spatial attention
        spatial1 = spatial_attention(en_ir, spatial_type)
        spatial2 = spatial_attention(en_vi, spatial_type)
        # get weight map, soft-max
        spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
        spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

        spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
        spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)
        tensor_f = spatial_w1 * en_ir + spatial_w2 * en_vi
        return tensor_f


# spatial attention
def spatial_attention(tensor, spatial_type='sum'):
    spatial = []
    if spatial_type is 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type is 'sum':
        spatial = tensor.sum(dim=1, keepdim=True)
    return spatial


# fuison strategy based on nuclear-norm (channel attention form NestFuse)
class Fusion_Nuclear(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        shape = en_ir.size()
        # calculate channel attention
        global_p1 = nuclear_pooling(en_ir)
        global_p2 = nuclear_pooling(en_vi)

        # get weight map
        global_p_w1 = global_p1 / (global_p1 + global_p2 + EPSILON)
        global_p_w2 = global_p2 / (global_p1 + global_p2 + EPSILON)

        global_p_w1 = global_p_w1.repeat(1, 1, shape[2], shape[3])
        global_p_w2 = global_p_w2.repeat(1, 1, shape[2], shape[3])

        tensor_f = global_p_w1 * en_ir + global_p_w2 * en_vi
        return tensor_f


# sum of S V for each chanel
def nuclear_pooling(tensor):
    shape = tensor.size()
    vectors = torch.zeros(1, shape[1], 1, 1).cuda()
    for i in range(shape[1]):
        u, s, v = torch.svd(tensor[0, i, :, :] + EPSILON)
        s_sum = torch.sum(s)
        vectors[0, i, 0, 0] = s_sum
    return vectors


class edge(nn.Module):
    def __init__(self, device, alpha_sal=0.7):
        super(edge, self).__init__()

        self.alpha_sal = alpha_sal
        self.mse_loss = torch.nn.MSELoss()
        self.laplacian_kernel = torch.tensor([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]], dtype=torch.float, requires_grad=False)
        self.laplacian_kernel = self.laplacian_kernel.view((1, 1, 3, 3))  # Shape format of weight for convolution
        self.laplacian_kernel = self.laplacian_kernel.to(device)

    def forward(self, y_pred):
        # Generate edge maps
        y_gt_edges = F.relu(torch.tanh(F.conv2d(y_pred, self.laplacian_kernel, padding=(1, 1))))
        sum=torch.sum(y_gt_edges)
        return sum

# Fusion strategy, two type
def features_grad(features):
    kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.cuda()
    _, c, _, _ = features.shape
    c = int(c)
    for i in range(c):
        feat_grad = F.conv2d(features[:, i:i + 1, :, :], kernel, stride=1, padding=1)
        if i == 0:
            feat_grads = feat_grad
        else:
            feat_grads = torch.cat((feat_grads, feat_grad), dim=1)
    return feat_grads


def vgf(over_a):
    feature_model = vgg16().cuda()
    feature_model.load_state_dict(torch.load('vgg16.pth'))
    feat_1 = torch.cat((over_a, over_a, over_a), dim=1)
    feat_1 = feature_model(feat_1)

    for i in range(len(feat_1)):
        m1 = torch.mean(features_grad(feat_1[i]).pow(2), dim=[1, 2, 3])
        if i == 0:
            w1 = torch.unsqueeze(m1, dim=-1)
        else:
            w1 = torch.cat((w1, torch.unsqueeze(m1, dim=-1)), dim=-1)
    weight_1 = torch.mean(w1, dim=-1) / 3500
    # weight_list = torch.cat((weight_1.unsqueeze(-1), weight_1.unsqueeze(-1)), -1)
    # weight_list = F.softmax(weight_list, dim=-1)

    return weight_1

def light_detect(uen_vi1):
    maskThreshold = 0.7
    luminance = uen_vi1 * uen_vi1
    luminance = np.where(luminance > maskThreshold, luminance, 0)
    mask = luminance > maskThreshold
    # uen_vi1=uen_vi1
    # 显示选区内原图
    uen_vi1[:, :][~mask] = 0
    # uen_vi[:, :, 1][~mask] = 0
    # uen_vi[:, :, 2][~mask] = 0
    uen_vi1 = uen_vi1 * 255
    uen_vi1 = uen_vi1.astype(np.uint8)
    cv2.imshow('light_detect',uen_vi1)
    cv2.waitKey(1)
    print(np.mean(uen_vi1))
    if np.mean(uen_vi1)>0:
        return False
    else:
        return True

def shalow_detect(uen_vi2):
    # 阴影选区
    maskThreshold = 0.95
    luminance = (1 - uen_vi2) * (1 - uen_vi2)
    luminance = np.where(luminance > maskThreshold, luminance, 0)
    mask = luminance > maskThreshold

    # 显示选区内原图
    uen_vi2[:, :][~mask] = 0
    uen_vi2[:, :][mask] = 1
    # uen_vi2[:, :, 1][~mask] = 0
    # uen_vi2[:, :, 2][~mask] = 0
    uen_vi2 = uen_vi2 * 255
    uen_vi2 = uen_vi2.astype(np.uint8)
    # cv2.imshow('shalow_detect',uen_vi2)
    # cv2.waitKey(1)
    # print(np.mean(uen_vi2))
    if np.mean(uen_vi2) > 0:
        return False
    else:
        return True

def Radio_detect(uen_ir3):
    maskThreshold = 0.7
    luminance = uen_ir3 * uen_ir3
    luminance = np.where(luminance > maskThreshold, luminance, 0)
    mask = luminance > maskThreshold

    # 显示选区内原图
    uen_ir3[:, :][~mask] = 0
    # uen_ir[:, :, 1][~mask] = 0
    # uen_ir[:, :, 2][~mask] = 0
    uen_ir3 = uen_ir3 * 255
    # kernel = np.ones((5, 5), np.uint8)  # 矩形结构
    # uen_ir3 = cv2.dilate(uen_ir3, kernel)  # 膨胀
    uen_ir3 = uen_ir3.astype(np.uint8)
    # cv2.imshow('Radio_detect',uen_ir3)
    # cv2.waitKey(1)
    print(np.mean(uen_ir3))
    if np.mean(uen_ir3) >0:
        return False
    else:
        return True

def get_entropy(img_):
    img_ = cv2.resize(img_, (100, 100)) # 缩小的目的是加快计算速度
    tmp = []
    for i in range(256):
        tmp.append(0)

    val = 0

    k = 0

    res = 0

    img = np.array(img_)

    for i in range(len(img)):

        for j in range(len(img[i])):

            val = img[i][j]

            tmp[val] = float(tmp[val] + 1)
            k =  float(k + 1)

    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)

    for i in range(len(tmp)):
            if(tmp[i] == 0):
                res = res
            else:
                res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
    return res

def brenner(img):
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-2):
        for y in range(0, shape[1]):
            out+=(int(img[x+2,y])-int(img[x,y]))**2
    out=out/(shape[0]*shape[1])
    return out

def AG_Entropy(a, c):

    a = a/np.max(a)*255
    a = a.astype(np.uint8)

    # cv2.imshow('a', a)
    # cv2.waitKey(0)

    c = c * 255
    c = c.astype(np.uint8)

    e3 = brenner(a)
    e4 = brenner(c)

    e1 = get_entropy(a)
    e2 = get_entropy(c)

    e5 = e3 / (e3+e4)+ e1 / (e1 + e2+0.0001)
    e6 = e4 / (e3+e4)+ e2 / (e1 + e2+0.0001)

    # print([e1, e2, e3, e4, e5, e6])
    if e5 < e6:
        return False
    else:
        return True

    return True


def brenner(img):
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-2):
        for y in range(0, shape[1]):
            out+=(int(img[x+2,y])-int(img[x,y]))**2
    out=out/(shape[0]*shape[1])
    return out

def AG_Entropyfusion(af):
    af = af / np.max(af) * 255
    af = af.astype(np.uint8)



    e3f = brenner(af)
    e1f = get_entropy(af)

    return e3f, e1f


class Fusion_strategytrain(nn.Module):
    def __init__(self):
        super(Fusion_strategytrain, self).__init__()
        self.fusion_add = Fusion_ADD()
        self.fusion_avg = Fusion_AVG()
        self.fusion_max = Fusion_MAX()
        self.fusion_spa = Fusion_SPA()
        self.fusion_nuc = Fusion_Nuclear()
        # self.edge1=edge(device=torch.device(device='cuda'))

    def forward(self, en_vi, en_ir):
        add=self.fusion_add(en_ir,en_vi)
        avg = self.fusion_avg(en_ir, en_vi)
        max = self.fusion_max(en_ir, en_vi)
        spa = self.fusion_spa(en_ir, en_vi)
        nu = self.fusion_nuc(en_ir, en_vi)
        t0 = np.random.choice(5, 1, False)
        t1 = t0[0]
        fev=[add, avg, max, spa, nu]
        return fev[t1]

class Fusion_strategytest(nn.Module):
    def __init__(self):
        super(Fusion_strategytest, self).__init__()
        self.fusion_add = Fusion_ADD()
        self.fusion_avg = Fusion_AVG()
        self.fusion_max = Fusion_MAX()
        self.fusion_spa = Fusion_SPA()
        self.fusion_nuc = Fusion_Nuclear()
        # self.edge1=edge(device=torch.device(device='cuda'))

    def forward(self, en_vi, en_ir):
        # add=self.fusion_add(en_ir,en_vi)
        # avg = self.fusion_avg(en_ir, en_vi)
        max = self.fusion_max(en_ir, en_vi)
        # spa = self.fusion_spa(en_ir, en_vi)
        # nu = self.fusion_nuc(en_ir, en_vi)
        # t0 = np.random.choice(5, 1, False)
        # t1 = t0[0]
        # fev=[add, avg, max, spa, nu]
        return max


class Fusion_strategy(nn.Module):
    def __init__(self):
        super(Fusion_strategy, self).__init__()
        self.fusion_add = Fusion_ADD()
        self.fusion_avg = Fusion_AVG()
        self.fusion_max = Fusion_MAX()
        self.fusion_spa = Fusion_SPA()
        self.fusion_nuc = Fusion_Nuclear()
        # self.edge1=edge(device=torch.device(device='cuda'))

    def forward(self, en_vi, en_ir):

        uen_vi = np.squeeze(en_vi.cpu().numpy(), axis=(0, 1))    #用于解码可见光图像，通过分类引入早退机制---高光、暗光
        uen_ir = np.squeeze(en_ir.cpu().numpy(), axis=(0, 1))    # 用于解码红外图像，通过分类引入早退机制---辐射特性

        # cv2.imshow('visible',  np.fabs(uen_vi ))
        # cv2.imshow('infrared', np.fabs(uen_ir))

        aa = uen_vi.copy()
        bb = uen_vi.copy()
        cc = uen_ir.copy()
        ddd = uen_vi.copy()
        eee = uen_ir.copy()

        #如何自动调节参数，尚未解决。。。
        caa = light_detect(aa)     #高光
        # cbb = shalow_detect(bb)    #暗光：新引入
        ccc = Radio_detect(cc)     #辐射特性
        cdd = AG_Entropy(ddd, eee)                         #梯度和信息熵

        # if  caa: #单独测试th1和th2 参数
        if caa&ccc&cdd:
            #早退机制
            print('early exit')
            return en_vi, 1
        else:
            print('later fusion')
            add = self.fusion_add(en_ir, en_vi)
            avg = self.fusion_avg(en_ir, en_vi)
            max = self.fusion_max(en_ir, en_vi)
            spa = self.fusion_spa(en_ir, en_vi)
            nu  = self.fusion_nuc(en_ir, en_vi)

            add1 = np.squeeze(add.cpu().numpy(), axis=(0, 1))
            avg1 = np.squeeze(avg.cpu().numpy(), axis=(0, 1))
            max1 = np.squeeze(max.cpu().numpy(), axis=(0, 1))
            spa1 = np.squeeze(spa.cpu().numpy(), axis=(0, 1))
            nu1 = np.squeeze(nu.cpu().numpy(), axis=(0, 1))
            #
            #
            addg, adde = AG_Entropyfusion(add1)
            avgg, avge = AG_Entropyfusion(avg1)
            maxg, maxe = AG_Entropyfusion(max1)
            spag, spae = AG_Entropyfusion(spa1)
            nug, nue = AG_Entropyfusion(nu1)
            #
            sumg = addg + avgg + maxg + spag + nug
            sume = adde + avge + maxe + spae + nue
            #
            addgg = addg / sumg
            avggg = avgg / sumg
            maxgg = maxg / sumg
            spagg = spag / sumg
            nugg = nug / sumg

            addee = adde / (sume + 0.0001) + addgg
            avgee = avge / (sume + 0.0001) + avggg
            maxee = maxe / (sume + 0.0001) + maxgg
            spaee = spae / (sume + 0.0001) + spagg
            nuee = nue / (sume + 0.0001) + nugg
            #
            emx = [addee, avgee, maxee, spaee, nuee]
            # print(emx)
            emx.sort(reverse=True)
            # print(emx)

            sf = avg
            if emx[0] == addee:
                sf = add
                print('add')
            if emx[0] == avgee:
                sf = avg
                print('avg')
            if emx[0] == maxee:
                sf = max
                print('max')
            if emx[0] == spaee:
                sf = spa
                print('spa')
            if emx[0] == nuee:
                sf = nu
                print('nu')


            return sf, 0





class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        # self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.conv_s1(hx)

        return xout

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src


### RSU-7 ###
class RSU7(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin


class DyReLU(nn.Module):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLU, self).__init__()
        self.channels = channels
        self.k = k
        self.conv_type = conv_type
        assert self.conv_type in ['1d', '2d']

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2*k)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.]*k + [0.5]*k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.]*(2*k - 1)).float())

    def get_relu_coefs(self, x):
        theta = torch.mean(x, axis=-1)
        if self.conv_type == '2d':
            theta = torch.mean(theta, axis=-1)
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x):
        raise NotImplementedError


class DyReLUA(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUA, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, 2*self.k) * self.lambdas + self.init_v
        # BxCxL -> LxCxBx1
        x_perm = x.transpose(0, -1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :self.k] + relu_coefs[:, self.k:]
        # LxCxBx2 -> BxCxL
        result = torch.max(output, dim=-1)[0].transpose(0, -1)

        return result


class DyReLUB(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUB, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k*channels)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, self.channels, 2*self.k) * self.lambdas + self.init_v

        if self.conv_type == '1d':
            # BxCxL -> LxBxCx1
            x_perm = x.permute(2, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # LxBxCx2 -> BxCxL
            result = torch.max(output, dim=-1)[0].permute(1, 2, 0)

        elif self.conv_type == '2d':
            # BxCxHxW -> HxWxBxCx1
            x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # HxWxBxCx2 -> BxCxHxW
            result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)

        return result


# def conv1x1(in_planes,out_planes,stride = 1):
#     return Dynamic_conv2d(in_planes,out_planes,kernel_size = 1,stride = stride,bias = False,)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# def conv3x3(in_planes,out_planes,stride = 1): # conv3x3 for dynamic convolution
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3,padding=1, stride=1, bias=False)


def conv3x3(in_planes,out_planes,stride = 1,groups = 1,dilation = 1): # conv3x3 for dynamic convolution
    return Dynamic_conv2d(in_planes,out_planes,kernel_size = 3,stride = stride,padding = dilation,groups = groups,bias = False,dilation = dilation)

def conv_1_2_hook(module, input, output):
    global vgg_conv1_2
    vgg_conv1_2 = output
    return None


def conv_2_2_hook(module, input, output):
    global vgg_conv2_2
    vgg_conv2_2 = output
    return None


def conv_3_3_hook(module, input, output):
    global vgg_conv3_3
    vgg_conv3_3 = output
    return None


def conv_4_3_hook(module, input, output):
    global vgg_conv4_3
    vgg_conv4_3 = output
    return None


def conv_5_3_hook(module, input, output):
    global vgg_conv5_3
    vgg_conv5_3 = output
    return None


class CPFE(nn.Module):
    def __init__(self, feature_layer=None, out_channels=32):
        super(CPFE, self).__init__()

        self.dil_rates = [3, 5, 7]

        # Determine number of in_channels from VGG-16 feature layer
        if feature_layer == 'conv5_3':
            self.in_channels = 512
        elif feature_layer == 'conv4_3':
            self.in_channels = 512
        elif feature_layer == 'conv3_3':
            self.in_channels = 3

        # Define layers
        self.conv_1_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.conv_dil_3 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[0], padding=self.dil_rates[0], bias=False)
        self.conv_dil_5 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[1], padding=self.dil_rates[1], bias=False)
        self.conv_dil_7 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[2], padding=self.dil_rates[2], bias=False)

        self.bn = nn.BatchNorm2d(out_channels*4)

    def forward(self, input_):
        # Extract features
        conv_1_1_feats = self.conv_1_1(input_)
        conv_dil_3_feats = self.conv_dil_3(input_)
        conv_dil_5_feats = self.conv_dil_5(input_)
        conv_dil_7_feats = self.conv_dil_7(input_)

        # Aggregate features
        concat_feats = torch.cat((conv_1_1_feats, conv_dil_3_feats, conv_dil_5_feats, conv_dil_7_feats), dim=1)
        bn_feats = F.relu(self.bn(concat_feats))

        return bn_feats

class ConvBlock(nn.Module):
    def __init__(self, inplane, outplane):
        super(ConvBlock, self).__init__()
        self.padding = (1, 1, 1, 1)
        self.conv = nn.Conv2d(inplane, outplane, kernel_size=3, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(outplane)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = F.pad(x, self.padding, 'replicate')
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.2)


class MiniInception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MiniInception, self).__init__()
        self.conv1_left  = ConvBnLeakyRelu2d(in_channels,   out_channels//2)
        self.conv1_right = ConvBnLeakyRelu2d(in_channels,   out_channels//2, padding=2, dilation=2)
        self.conv2_left  = ConvBnLeakyRelu2d(out_channels,  out_channels//2)
        self.conv2_right = ConvBnLeakyRelu2d(out_channels,  out_channels//2, padding=2, dilation=2)
        self.conv3_left  = ConvBnLeakyRelu2d(out_channels,  out_channels//2)
        self.conv3_right = ConvBnLeakyRelu2d(out_channels,  out_channels//2, padding=2, dilation=2)
    def forward(self,x):
        x = torch.cat((self.conv1_left(x), self.conv1_right(x)), dim=1)
        x = torch.cat((self.conv2_left(x), self.conv2_right(x)), dim=1)
        x = torch.cat((self.conv3_left(x), self.conv3_right(x)), dim=1)
        return x

def shuffle_channels(x, groups):
    """shuffle channels of a 4-D Tensor"""
    batch_size, channels, height, width = x.size()
    assert channels % groups == 0
    channels_per_group = channels // groups
    # split into groups
    x = x.view(batch_size, groups, channels_per_group,
               height, width)
    # transpose 1, 2 axis
    x = x.transpose(1, 2).contiguous()
    # reshape into orignal
    x = x.view(batch_size, channels, height, width)
    return x


class Fire(nn.Module):
    def __init__(self, in_channel, out_channel, squzee_channel):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channel, squzee_channel, 1,groups=1),
            nn.BatchNorm2d(squzee_channel),
            nn.ReLU(inplace=True)
        )
        self.expand_1x1 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 1,groups=16),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )
        self.expand_3x3 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 3, padding=1,groups=16),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = torch.cat([
            self.expand_1x1(x),
            self.expand_3x3(x)
        ], 1)
        return x

class CA(nn.Module):
    def __init__(self,in_ch):
        super(CA, self).__init__()
        self.avg_weight = nn.MaxPool2d(1)
        self.max_weight = nn.MaxPool2d(1)
        self.fus = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(in_ch // 2, in_ch, 1, 1, 0),
        )
        self.c_mask = nn.Sigmoid()
    def forward(self, x):
        avg_map_c = self.avg_weight(x)
        max_map_c = self.max_weight(x)
        c_mask = self.c_mask(torch.add(self.fus(avg_map_c), self.fus(max_map_c)))
        return torch.mul(x, c_mask)


class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation,
                            norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)

class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)

        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):

        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H),
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2,
                                                                                       self.kernel_size,
                                                                                       self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.group_planes // 2, self.group_planes // 2,
                                                             self.group_planes], dim=0)
        # pdb.set_trace()

        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)

        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        # stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        # nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


class AxialBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride,
                                          width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print(out.shape)
        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out

# Convolution operation
class f_ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(f_ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        #self.batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        #out = self.batch_norm(out)
        out = F.relu(out, inplace=True)
        return out

def fusion_channel_sf(f1, kernel_radius=5):
    """
    Perform channel sf fusion two features
    """
    device = f1.device
    b, c, h, w = f1.shape
    r_shift_kernel = torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]) \
        .cuda(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
    b_shift_kernel = torch.FloatTensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]]) \
        .cuda(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
    f1_r_shift = F.conv2d(f1, r_shift_kernel, padding=1, groups=c)
    f1_b_shift = F.conv2d(f1, b_shift_kernel, padding=1, groups=c)
    # f2_r_shift = f.conv2d(f2, r_shift_kernel, padding=1, groups=c)
    # f2_b_shift = f.conv2d(f2, b_shift_kernel, padding=1, groups=c)

    f1_grad = torch.pow((f1_r_shift - f1), 2) + torch.pow((f1_b_shift - f1), 2)
    # f2_grad = torch.pow((f2_r_shift - f2), 2) + torch.pow((f2_b_shift - f2), 2)

    kernel_size = kernel_radius * 2 + 1
    add_kernel = torch.ones((c, 1, kernel_size, kernel_size)).float().cuda(device)
    kernel_padding = kernel_size // 2
    f1_sf = torch.sum(F.conv2d(f1_grad, add_kernel, padding=kernel_padding, groups=c), dim=1)
    # f2_sf = torch.sum(f.conv2d(f2_grad, add_kernel, padding=kernel_padding, groups=c), dim=1)
    # weight_zeros = torch.zeros(f1_sf.shape).cuda(device)
    # weight_ones = torch.ones(f1_sf.shape).cuda(device)

    # get decision map
    # dm_tensor = torch.where(f1_sf > f2_sf, weight_ones, weight_zeros).cuda(device)
    # dm_np = dm_tensor.squeeze().cpu().numpy().astype(np.int)
    return f1_sf

class SODModel(nn.Module):
    def __init__(self):
        super(SODModel, self).__init__()
        self.Fire = conv3x3(17, 64)
        # self.Fire=Fire(6,64,64)
        # self.Fire1 = Fire(64, 64, 64)
        # self.conv4 = ConvBlock(64, 1)
        self.conv1 = conv3x3(64, 64)
        self.conv2 = conv3x3(64, 32)
        self.conv3 = conv3x3(32, 1)
        #
        # self.Fire1 = conv3x3(17, 64)
        # self.Fire=Fire(6,64,64)
        # self.Fire1 = Fire(64, 64, 64)
        # self.conv4 = ConvBlock(64, 1)
        # self.conv11 = conv3x3(64, 64)
        # self.conv21 = conv3x3(64, 34)
        self.conv31 = conv3x3(32, 1)
        self.conv33 = conv3x3(32, 1)

        self.Firef = conv3x3(3, 32)
        # self.Fire=Fire(6,64,64)
        # self.Fire1 = Fire(64, 64, 64)
        # self.conv4 = ConvBlock(64, 1)
        # self.conv1f = conv3x3(64, 64)
        self.conv2f = conv3x3(32, 16)
        self.conv3f = conv3x3(16, 1)
        self.relu = nn.ReLU(inplace=True)
        # self.axial_attng1 = AxialBlock(32, 32 // 2, kernel_size=400)
        # self.axial_attng2 = AxialBlock(32, 32 // 2, kernel_size=400)
        #
        # self.axial_attn = AxialBlock(192, 192 // 2, kernel_size=400)

        # self.axial_fusion = nn.Sequential(f_ConvLayer(2 * channels, channels, 1, 1))

        # self.lrsu=RSU7(32,16,32)
        # self.convl = ConvBlock(32, 1)
        # self.rrsu=RSU7(17,128,32)
        # self.convr = ConvBlock(32, 1)
        # self.ca=CA(32)
        # self.fusiontest = Fusion_strategytest()#SLDR test
        self.fusion=Fusion_strategy()
        self.fusion_avg1 = Fusion_AVG()
        # self.fusiontrain=Fusion_strategytrain()
        # self.frsu = RSU7(4,256,64)
        # self.convf = ConvBlock(64, 1)

        # self.relu = DyReLUB(10, conv_type='2d')

    def forward(self, input_,input_np,dg1,dg2,g1,g2,inpd,irt):

        # l1=self.lrsu(g1)
        # f7=self.convl(l1)
        #
        # l2=self.lrsu(g2)
        # f77 = self.convl(l2)
        # l2=self.rrsu(g2)
        # f77 = self.convl(l2)
        #
        # f=self.fusion(f7,f77)
        # f12 = self.frsu(f)
        # f12 = self.convl(f12)

        ################################################################
        # f3 = torch.cat((input_, input_), dim=1)


        f4=self.Fire(g1)
        f5 = self.conv1(f4)
        f6 = self.conv2(f5)
        # f6 = self.axial_attng1(f6)
        f71 = self.conv3(f6)
        f7=self.relu(f71)

        # f44=self.Fire(g2)  #g1 visible image  f77  ; g2 infrared image f7
        # f55 = self.conv1(f44)
        # f66 = self.conv2(f55)
        # # f66 = self.axial_att
        # ng1(f66)
        # f72 = self.conv3(f66)
        # f77 = self.relu(f72)

        # # f33 = torch.cat((input_np, input_np), dim=1)

        f44=self.Fire(g2)  #g1 visible image  f77  ; g2 infrared image f7
        f55 = self.conv1(f44)
        f66 = self.conv2(f55)
        # f66 = self.axial_attng1(f66)
        f72 = self.conv31(f66)
        f77 = self.relu(f72)

        # # f8 = torch.cat((f77, g2, f7, g1), dim=1)
        # f8 = torch.cat((f7,g1,f77,g2), dim=1)
        # a_cat = torch.cat([self.axial_attn(f6), self.axial_attn(f66)], 1)
        # a_init = self.axial_fusion(a_cat)
        # tmpv = f7.repeat(1, 32, 1, 1)
        # tmpir = f77.repeat(1,32,1,1)


        # tmpir = f7.repeat(1, 17, 1, 1)
        # tmpvr = f77.repeat(1, 17, 1, 1)

        ffff, et = self.fusion(f7, f77) #动态融合方法:受启发于DConv and DCNN

        # ffff=self.fusion_avg1(f7,f77) #常规单一融合方法:ADD MAX, NU, AVG,....
        # et=0

        # ffff = self.fusiontest(f7, f77)

        # f1 = self.conv3(fadd)
        # f2 = self.relu(f1)
        # f8=(ffff+f7+f77)/3


        # f8 = ffff.repeat(1, 17, 1, 1)
        # # f8 = torch.cat((f8, f7, f77), dim=1)
        # tmpir=self.Fire(f8)
        # tmpir = self.conv1(tmpir)
        # tmpir = self.conv2(tmpir)
        # f12 = self.conv31(tmpir)


        # f8=self.ca(f8)
        # f88=self.axial_attn(f8)
        # ffff = self.fusiontrain(inpd, irt)


        #one method 早退的性能仅仅依赖于复原模块性能
        # if et:
        #     return (f7, f77, ffff)
        # else:
        #     f8 = torch.cat((ffff, f7, f77), dim=1)
        #     f9 = self.Firef(f8)
        #     f11 = self.conv2f(f9)
        #     f12 = self.conv3f(f11)
        #     f12 = self.relu(f12)
        #     return (f7, f77, f12)

        #the second method 早退的性能进一步增强
        if et:
            f8 = torch.cat((ffff, ffff, ffff), dim=1)
            f9 = self.Firef(f8)
            f11 = self.conv2f(f9)
            f12 = self.conv3f(f11)
            f12 = self.relu(f12)
            return (f7, f77, f12)
        else:
            f8 = torch.cat((ffff, f7, f77), dim=1)
            f9 = self.Firef(f8)
            f11 = self.conv2f(f9)
            f12 = self.conv3f(f11)
            f12 = self.relu(f12)
            return (f7, f77, f12)

        # f4 = shuffle_channels(f4, 16)
        # f4=self.Fire1(f4)
        # f4 = shuffle_channels(f4, 16)

        # x_rgb1, _ = self.c3(f5)
        # x_rgb = torch.mul(f4, x_rgb1)

        #
        # f8=f7+x_rgb
        # f9=shuffle_channels(f4,2)
        # f2 = self.conv3(f9)
        # f2=f2+f8
        # f9 = shuffle_channels(f2, 2)

        # f3 = self.conv4(f4)
        ################################################################

        # out1 = self.group_conv1(input_)
        # out2 = self.group_conv2(input_np)
        # # f8 = torch.cat((out1, out2), dim=1)
        # f8=out1+out2
        # out = shuffle_channels(f8, 4)
        # out = self.depthwise_conv3(out)
        # out = self.bn2(out)
        # f8 = out+out1+out2
        # out = shuffle_channels(f8, 4)
        # f3 = self.group_conv5(out)
        ################################################################
        # f3 = self.conv5(input_)
        # f4 =self.conv6(input_np)
        # f7 = torch.cat((f3, f4), dim=1)
        # f9 = shuffle_channels(f7, 2)
        # # f8 = self.conv2(f9)
        # # f10=self.depthwise_conv3(f8)
        # # f7 = torch.cat((f10, f7), dim=1)
        # # f11 = shuffle_channels(f8, 2)
        # f3 = self.conv4(f9)
        ################################################################
        # Process high level features
        # f3 = self.conv5(input_)
        # f4 =self.conv6(input_np)
        # f7 = torch.cat((f3, f4), dim=1)
        # f9 = shuffle_channels(f7, 2)
        # f8 = self.conv2(f9)
        # f10=self.depthwise_conv3(f8)
        # f7 = torch.cat((f10, f7), dim=1)
        # f11 = shuffle_channels(f7, 2)
        # f3 = self.conv4(f11)
        ################################################################
        # f3 = self.conv5(input_)
        # f4 =self.conv5(input_np)
        # f7 = torch.cat((f3, f4), dim=1)
        # f8 = self.conv6(f7)
        # f10=self.depthwise_conv3(f8)
        # f11=self.depthwise_conv3(f8)
        # f12 = torch.cat((f10, f7,f11), dim=1)
        # f13 = shuffle_channels(f12, 3)
        # f3 = self.conv4(f13)
        # return (f7)


def test():
    dummy_input = torch.randn(2, 3, 256, 512)

    model = SODModel()
    out, ca_act_reg = model(dummy_input)

    print(model)
    print('\nModel input shape :', dummy_input.size())
    print('Model output shape :', out.size())
    print('ca_act_reg :', ca_act_reg)


class Discriminator_v(nn.Module):
    def __init__(self):
        super(Discriminator_v, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(512, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x1 = self.net(x).view(batch_size, 1)
        return torch.sigmoid(x1)

class Discriminator_i(nn.Module):
    def __init__(self):
        super(Discriminator_i, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(512, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x1 = self.net(x).view(batch_size,1)
        return torch.sigmoid(x1)

class DDcGAN(nn.Module):
    def __init__(self, if_train=False):
        super(DDcGAN, self).__init__()
        self.if_train=if_train

        self.G=SODModel()
        self.Dv=Discriminator_v()
        self.Di=Discriminator_i()
        # self.down=nn.Sequential(
        #     nn.AvgPool2d(3,2,1),
        #     nn.AvgPool2d(3,2,1))

    def forward(self,input_,input_np,dg1,dg2,g1,g2,inpd,irt):
        e7,e77,fusion_v=self.G(input_,input_np,dg1,dg2,g1,g2,inpd,irt)
        # image_save(fusion_v[0:1,:,:,:],'./test/'+str(len(os.listdir('./test')))+'.jpg')
        # image_save(fusion_v[1:2,:,:,:],'./test/'+str(len(os.listdir('./test')))+'.jpg')
        if self.if_train:
            score_v=self.Dv(e7)
            score_i=self.Di(e77)
            score_Gv=self.Dv(fusion_v)
            score_Gi=self.Di(fusion_v)
            return fusion_v,fusion_v,score_v,score_i,score_Gv,score_Gi
        else:
            return e7,e77,fusion_v


if __name__ == '__main__':
    vis = torch.rand((1, 1, 256, 256))
    ir = torch.rand((1, 1, 64, 64))
    model = Discriminator_i()
    output = model(ir)
    test()
