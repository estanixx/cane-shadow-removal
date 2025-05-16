from utils import display_results, images
from methods import process_image_hsv, remove_shadows_medianblur_lab_color, remove_shadows_morph_gray, shadow_removal
import cv2
import numpy as np

def increase_brightness_otsu(img_bgr, factor):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Usar el canal V (Value) para la máscara y modificación, según tu código.
    # El índice 2 corresponde al canal V en HSV.
    v_channel_for_mask = img_hsv[:, :, 2]

    _, binary_mask_single_channel = cv2.threshold(
        v_channel_for_mask, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    binary_mask_single_channel = cv2.GaussianBlur(binary_mask_single_channel, (15, 15), 0)
    condition_met = (binary_mask_single_channel == 255)

    # Modificar el canal V: multiplicar por 2 donde la condición se cumple.
    # Convertir a np.int16 para evitar desbordamiento (overflow) durante la multiplicación.
    v_channel = img_hsv[:, :, 2]
    brighten_image = v_channel * factor
    v_channel_modified = np.where(condition_met, brighten_image, v_channel).astype('uint8')
    # Asegurar que los valores estén en el rango [0, 255] después de la multiplicación y antes de reconvertir a uint8.
    v_channel_modified =  np.clip(v_channel_modified, 0, 255).astype(np.uint8)
    # v_channel_modified = cv2.equalizeHist(v_channel_modified)
    img_hsv[:, :, 2] = v_channel_modified
    img_hsv[:, :, 1] = (img_hsv[:, :, 1] * 0.9).astype('uint8')
    output_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return output_bgr
    

def pipeline1(img):
    img = increase_brightness_otsu(img, 1.5)
    img = remove_shadows_medianblur_lab_color(img, blur_ksize=51)
    #img = decrease_brightness_otsu(img, 0.9)
    
    img = cv2.medianBlur(img, 5, 0)

    return img

display_results(
    pipeline1,
    dict(list(images.items())[7:]),
)
