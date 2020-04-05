from __future__ import division
import torch
import torch.nn as nn

import os
import time
import numpy as np

from imageio import imread
import PIL.Image
from resblock import resblock,conv_bn_relu_res_block
from utils import save_matv73,reconstruction,load_mat,mrae,rmse

model_path = './models/HS_veins_34to1band.pkl'
img_path = '../dataset/veins_t34bands/test_data/rgb/'
result_path = '../dataset/veins_t34bands/test_data/inference/'
gt_path = '../dataset/veins_t34bands/test_data/mat/'
var_name = 'rad'

save_point = torch.load(model_path)
model_param = save_point['state_dict']
model = resblock(conv_bn_relu_res_block,10,3,1)
model.load_state_dict(model_param)

model = model.cuda()
model.eval()



for img_name in sorted(os.listdir(img_path)):
    img_path_name = os.path.join(img_path, img_name)
    rgb = imread(img_path_name)   
    rgb = rgb/255
    rgb = np.expand_dims(np.transpose(rgb,[2,1,0]), axis=0).copy() 
   
    img_res1 = reconstruction(rgb,model)
    img_res2 = np.flip(reconstruction(np.flip(rgb, 2).copy(),model),1)
    img_res3 = (img_res1+img_res2)/2

    mat_name = img_name[:-4] + '.mat'
    mat_dir= os.path.join(result_path, mat_name)

    save_matv73(mat_dir, var_name,img_res3)

    gt_name =  img_name[12:-4] + '.mat'
    gt_dir= os.path.join(gt_path, gt_name)
    gt = load_mat(gt_dir,var_name)
    mrae_error =  mrae(img_res3, gt['rad'][:,:,1])
    rrmse_error = rmse(img_res3, gt['rad'][:,:,1])
    print("[%s] MRAE=%0.9f RRMSE=%0.9f" %(img_name,mrae_error,rrmse_error))
