from utils import display_results, get_jd_images, get_moving_case_images, get_static_case_images
from methods import remove_shadows_medianblur_lab_color, remove_shadows_morph_gray, decrease_brightness_otsu_hsv, increase_brightness_otsu
import cv2
import numpy as np



def pipeline1(img):
    """Removes shadows"""
    img = increase_brightness_otsu(img, 1.5)
    img = remove_shadows_medianblur_lab_color(img, blur_ksize=51)
    img = decrease_brightness_otsu_hsv(img, 0.70)
    return img

def pipeline2(img):
    """Removes shadows and return an outline"""
    img = pipeline1(img)
    img = remove_shadows_morph_gray(img)
    return img

display_results(
    pipeline1,
    dict(list(get_jd_images().items())[:4]),
)
