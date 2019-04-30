import skimage
from skimage import measure
import numpy as np

def psnr(im1, im2):
    def im2double(im):
        min_val, max_val = 0, 255
        out = (im.astype(np.float64)-min_val) / (max_val-min_val)
        return out
    im1 = im2double(im1)
    im2 = im2double(im2)
    psnr = skimage.measure.compare_psnr(im1, im2, data_range=1)
    return psnr
