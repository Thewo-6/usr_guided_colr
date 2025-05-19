
import argparse
import matplotlib.pyplot as plt
import numpy as np 
import torch
import cv2

from colorizers import *

parser = argparse.ArgumentParser()
parser.add_argument('-i','--img_path', type=str, default='imgs/ansel_adams3.jpg')
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
parser.add_argument('-o','--save_prefix', type=str, default='saved', help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
opt = parser.parse_args()

# Load hint map and mask
ab_hint_np = np.load('ab_hint_map.npy')       # shape: (H, W, 2)
mask_hint_np = np.load('hint_mask.npy')       # shape: (H, W)

# Resizing op
ab_hint_resized = cv2.resize(ab_hint_np, (256, 256), interpolation=cv2.INTER_NEAREST)
mask_hint_resized = cv2.resize(mask_hint_np, (256, 256), interpolation=cv2.INTER_NEAREST)

# Convert to torch tensors
ab_hint_tensor = torch.from_numpy(ab_hint_resized).permute(2, 0, 1).unsqueeze(0).float()  # shape: (1, 2, H, W)
mask_hint_tensor = torch.from_numpy(mask_hint_resized).unsqueeze(0).unsqueeze(0).float()  # shape: (1, 1, H, W)

# load colorizers
#colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()
if(opt.use_gpu):
	#colorizer_eccv16.cuda()
	colorizer_siggraph17.cuda()

# default size to process images is 256x256
# grab L channel in both original ("orig") and resized ("rs") resolutions
img = load_img(opt.img_path)
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
if(opt.use_gpu):
	tens_l_rs = tens_l_rs.cuda()

# colorizer outputs 256x256 ab map
# resize and concatenate to original L channel
img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
#out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
out_ab = colorizer_siggraph17(tens_l_rs, input_B=ab_hint_tensor, mask_B=mask_hint_tensor)
out_img_siggraph17 = postprocess_tens(tens_l_orig, out_ab.cpu())

#plt.imsave('%s_eccv16.png'%opt.save_prefix, out_img_eccv16)
plt.imsave('%s_siggraph17.png'%opt.save_prefix, out_img_siggraph17)

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(img_bw)
plt.title('Input')
plt.axis('off')

""" plt.subplot(2,2,3)
plt.imshow(out_img_eccv16)
plt.title('Output (ECCV 16)')
plt.axis('off') """

plt.subplot(2,2,4)
plt.imshow(out_img_siggraph17)
plt.title('Output (SIGGRAPH 17)')
plt.axis('off')
plt.show()
