"""
# > Script for measuring quantitative performances in terms of
#    - Structural Similarity Metric (SSIM) 
#    - Peak Signal to Noise Ratio (PSNR)
# > Maintainer: https://github.com/xahidbuffon
"""
## python libs
import numpy as np
from PIL import Image, ImageOps
from glob import glob
from os.path import join
from ntpath import basename
## local libs
from uqim_utils import getUIQM


def measure_UIQMs(dir_name, im_res=(256, 256)):
    paths = sorted(glob(join(dir_name, "*.*")))
    uqims = []
    i=0
    for img_path in paths:
        print(i)
        i=i+1
        im = Image.open(img_path).resize(im_res)
        im_array = np.array(im)
        # print(im_array)
        uiqm = getUIQM(im_array)
        uqims.append(uiqm)
    return np.array(uqims)

"""
Get datasets from
 - http://irvlab.cs.umn.edu/resources/euvp-dataset
 - http://irvlab.cs.umn.edu/resources/ufo-120-dataset
"""
# #inp_dir = "/home/xahid/datasets/released/EUVP/test_samples/Inp/"
# inp_dir = "/home/mbzirc/Downloads/AhsanBB/Dehazing/UEIB_Data/UEIB_Dataset/Data1/test/testA"
# # UIQMs of the distorted input images 
# inp_uqims = measure_UIQMs(inp_dir)
# print ("Input UIQMs >> Mean: {0} std: {1}".format(np.mean(inp_uqims), np.std(inp_uqims)))

# UIQMs of the enhanceded output images
# gen_dir = "eval_data/euvp_test/funie-gan/" 
gen_dir = "/home/mbzirc/Downloads/AhsanBB/Dehazing/UEIB_Data/Reference_papers/Comparison_Results/11-Funie-GAN/PyTorch/UGAN_output/UCCS" 
# gen_dir = "/home/mbzirc/Downloads/AhsanBB/Dehazing/UEIB_Data/data/Challenging_Data/output_Fusion" 
gen_uqims = measure_UIQMs(gen_dir)
print ("Enhanced UIQMs >> Mean: {0} std: {1}".format(np.mean(gen_uqims), np.std(gen_uqims)))



