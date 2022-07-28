from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
import cv2
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from subprocess import Popen, PIPE
from model import SODModel
from dataloader import InfDataloader, SODLoader
from torchvision import transforms
# from thop import profile
##D:\software\Python36\mydataset\DST/400/IR
##D:\software\Python36\mydataset\DST/400/VIS
##
import re

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters to train your model.')
    parser.add_argument('--imgs_folder', default='G:\mydataset\DST/data/VIFB/VIS', help='Path to folder containing images', type=str)
    parser.add_argument('--imgs_folderi', default='G:\mydataset\DST/data/VIFB/IR',
                        help='Path to folder containing images', type=str)
    parser.add_argument('--imgs_folderg', default='G:\mydataset\DST//data/VIFB/VIS',
                        help='Path to folder containing images', type=str)
    parser.add_argument('--model_path', default='./models/best_epoch-16_mae-0.0994_loss-0.0909.pth', help='Path to model', type=str)
    parser.add_argument('--use_gpu', default=True, help='Whether to use GPU or not', type=bool)
    parser.add_argument('--img_size', default=400, help='Image size to be used', type=int)
    parser.add_argument('--bs', default=24, help='Batch Size for testing', type=int)

    return parser.parse_args()

def guideFilter(I, p, winSize, eps):

    mean_I = cv2.blur(I, winSize)      # I的均值平滑
    mean_p = cv2.blur(p, winSize)      # p的均值平滑

    mean_II = cv2.blur(I * I, winSize) # I*I的均值平滑
    mean_Ip = cv2.blur(I * p, winSize) # I*p的均值平滑

    var_I = mean_II - mean_I * mean_I  # 方差
    cov_Ip = mean_Ip - mean_I * mean_p # 协方差

    a = cov_Ip / (var_I + eps)         # 相关因子a
    b = mean_p - a * mean_I            # 相关因子b

    mean_a = cv2.blur(a, winSize)      # 对a进行均值平滑
    mean_b = cv2.blur(b, winSize)      # 对b进行均值平滑

    q = mean_a * I + mean_b
    return q

def run_inference(args):
    # Determine device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(device='cuda')
    else:
        device = torch.device(device='cpu')

    # Load model
    model = SODModel()
    chkpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(chkpt['model'])
    model.to(device)
    model.eval()
    print(model)

    inf_data = InfDataloader(img_folder=args.imgs_folder, im=args.imgs_folderi,im1=args.imgs_folderg,target_size=args.img_size)
    # Since the images would be displayed to the user, the batch_size is set to 1
    # Code at later point is also written assuming batch_size = 1, so do not change
    inf_dataloader = DataLoader(inf_data, batch_size=1, shuffle=False, num_workers=2)
    i=0
    print("Press 'q' to quit.")

    with torch.no_grad():
        for batch_idx, (img_np, img_tor1,img_np1,img_torr,img_np2,img_togt,inp_img11,IR_img11) in enumerate(inf_dataloader, start=1):

            # img_tord1 = v.to(device)
            # img_tordr = ii.to(device)
            img_np = img_np.to(device)
            img_tor1 = img_tor1.to(device)

            inp_img11 = inp_img11.to(device)
            IR_img11 = IR_img11.to(device)
            # img_togt  =img_togt.to(device)
            # d=inf_dataloader.dataset.im_paths[i]

            # f=d[20:]
            # R = img_tord1[0][0]
            # G = img_tord1[0][1]
            # B = img_tord1[0][2]
            # A = 0.299 * R + 0.587 * G + 0.114 * B
            # A = A.view(1,1, 400, 400)
            #
            # R1 = img_tordr[0][0]
            # G1 = img_tordr[0][1]
            # B1 = img_tordr[0][2]
            # A1 = 0.299 * R1 + 0.587 * G1 + 0.114 * B1
            # A1 = A1.view(1, 1, 400, 400)

            # G = tensor[1]
            # B = tensor[2]
            # tensor[0] = 0.299 * R + 0.587 * G + 0.114 * B
            # img_tord1 = transforms.Compose([
            #     transforms.Grayscale(num_output_channels=1),  # 彩色图像转灰度图像num_output_channels默认1
            #     transforms.ToTensor()
            # ])
            # img_tordr = transforms.Compose([
            #     transforms.Grayscale(num_output_channels=1),  # 彩色图像转灰度图像num_output_channels默认1
            #     transforms.ToTensor()
            # ])
            # degradation_map = A1.repeat(A1.shape[0], 1, 128, 128)
            # degradation_map1 = A.repeat(A.shape[0], 1, 128, 128)
            # degradation_map = degradation_map.to(self.device)
            v1, i1, la = model(img_np, img_tor1, inp_img11, IR_img11, inp_img11, IR_img11,inp_img11,IR_img11)
            # _,_,pred_masks=model(A1, A,A1,A,A,A1)

            # flops, params = profile(model, inputs=(img_tordr,img_tord1))
            # print(flops,params)
            # img_tor = img_tor1.to(device)
            # pred_maskse, _ = model(img_tor)

            # pred_masks = pred_masks
            # Assuming batch_size = 1
            # img_np = np.squeeze(img_np.numpy(), axis=0)
            # img_np = img_np.astype(np.uint8)
            # img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            pred_masks_raw = np.squeeze(la.cpu().numpy(), axis=(0, 1))
            # pred_masks_round = np.squeeze(pred_masks.round().cpu().numpy(), axis=(0, 1))

            # print('Image :', batch_idx)
            #
            # cv2.imshow('Input Image', img_np)
            # cv2.imshow('Generated Saliency Mask', pred_masks_raw)
            # cv2.imshow('Rounded-off Saliency Mask', pred_masks_round)
            #
            #
            # key = cv2.waitKey(0)
            # if key == ord('q'):
            #     break
            #Ir_attention
            #Ir_attention_fusion
            # cv2.imwrite('output/Ir_attention_fusion/2.png', np.fabs(pred_masks_raw / pred_masks_raw.max() * 255))
            # eps = 0.001
            # winSize = (2, 2)
            #
            # I = pred_masks_raw/pred_masks_raw.max()  # 将图像归一化
            # p = I
            # guideFilter_img = guideFilter(I, p, winSize, eps)
            #
            # # 保存导向滤波结果
            # guideFilter_img = guideFilter_img * 255
            # guideFilter_img[guideFilter_img > 255] = 255
            # guideFilter_img = np.round(guideFilter_img)
            # guideFilter_img = guideFilter_img.astype(np.uint8)
            #
            # cv2.imshow("winSize_5", guideFilter_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # cv2.imwrite('output/Ir_attention_fusion/%s.png' % i, guideFilter_img)

            tp=inf_dataloader.dataset.im_paths[i][29:]
            print(tp)
            # cv2.imwrite('output/Ir_attention_fusion/' + tp, np.fabs(pred_masks_raw * 255.0))
            cv2.imwrite('D:\Ablation study/VIFB\T1/8/'+tp,np.fabs(pred_masks_raw*255.0))
            i = i + 1
            if i==2:
                start = time.time()
            if i==1002:
                break

        end=time.time()
        print((end-start)/300.0*60)

def calculate_mae(args):
    # Determine device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(device='cpu')
    else:
        device = torch.device(device='cpu')

    # Load model
    model = SODModel()
    chkpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(chkpt['model'])
    model.to(device)
    model.eval()

    test_data = SODLoader(mode='test', augment_data=False, target_size=args.img_size)
    test_dataloader = DataLoader(test_data, batch_size=args.bs, shuffle=False, num_workers=2)

    # List to save mean absolute error of each image
    mae_list = []
    with torch.no_grad():
        for batch_idx, (inp_imgs, gt_masks) in enumerate(tqdm.tqdm(test_dataloader), start=1):
            inp_imgs = inp_imgs.to(device)
            gt_masks = gt_masks.to(device)
            pred_masks, _ = model(inp_imgs)

            mae = torch.mean(torch.abs(pred_masks - gt_masks), dim=(1, 2, 3)).cpu().numpy()
            mae_list.extend(mae)

    print('MAE for the test set is :', np.mean(mae_list))


if __name__ == '__main__':
    rt_args = parse_arguments()
    # calculate_mae(rt_args)
    run_inference(rt_args)


    # thermal = "F:\PycharmProjects\lena/venv/PyTorch-Pyramid-Feature-Attention-Network-for-Saliency-Detection2/firefly/darknet detector test cfg/thermal.data cfg/yolov3-spp-custom.cfg yolov3-spp-thermal.weights -dont_show"
    # rgb = "./firefly/darknet detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights -dont_show"
    #
    # pthermal = Popen([thermal], shell=True, stdout=PIPE, stdin=PIPE)
    # prgb = Popen([rgb], shell=True, stdout=PIPE, stdin=PIPE)
    #
    # # img_type, img_path = raw_input("Enter image type(thermal/rgb) and image path").split(" ")
    # # img_path = img_path + '\n'
    # img_type="thermal"
    # img_path = bytes('1061', 'UTF-8')  # Needed in Python 3.
    #
    # result = None
    # if img_type == "thermal":
    #     pthermal.stdin.write(img_path)
    #     pthermal.stdin.flush()
    #     result = pthermal.stdout.readline().strip()
    # if img_type == "rgb":
    #     prgb.stdin.write(img_path)
    #     prgb.stdin.flush()
    #     result = prgb.stdout.readline().strip()
    #
    # print(result)
