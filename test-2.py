from utils import display_results, images
from methods import process_image_hsv, remove_shadows_medianblur_lab_color, remove_shadows_morph_gray, shadow_removal
import cv2
import numpy as np


    

def pipeline2(img):
    img = remove_shadows_medianblur_lab_color(process_image_hsv(img), blur_ksize=17)
    if len(img.shape) == 3 and img.shape[2] == 3: # Color image
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y_channel, cr_channel, cb_channel = cv2.split(img_ycrcb)
        cv2.equalizeHist(y_channel, y_channel) # Apply to Y (luminance) channel
        equalized_ycrcb = cv2.merge((y_channel, cr_channel, cb_channel))
        img = cv2.cvtColor(equalized_ycrcb, cv2.COLOR_YCrCb2BGR)
    else: # Grayscale image
        img = cv2.equalizeHist(img)
    # img = remove_shadows_medianblur_lab_color(img, blur_ksize=61)
    # img = decrease_brightness_otsu(img, 0.9)
    
    # img = cv2.medianBlur(equalized_img, 9, 0)

    return img

display_results(
    pipeline2 ,
    dict(list(images.items())[1:5]),
)