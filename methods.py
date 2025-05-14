import cv2
import numpy as np
def remove_shadows_medianblur_lab_color(img_bgr, blur_ksize=21):
    """
    Elimina sombras estimando el fondo en el canal L (LAB) usando MedianBlur
    y normalizando la luminancia. Devuelve una imagen a color.

    Args:
        img_bgr: Imagen de entrada en formato BGR.
        blur_ksize: Tamaño del kernel para cv2.medianBlur (debe ser impar y > 1).
                    Ajustar según el tamaño de la sombra/detalle.

    Returns:
        Imagen procesada en BGR con sombras reducidas y color preservado.
    """
    try:
        # Asegurar que el kernel sea impar y mayor que 1
        if blur_ksize <= 1 or blur_ksize % 2 == 0:
            print(f"Advertencia: blur_ksize debe ser impar y > 1. Ajustando a {max(3, blur_ksize + (1 if blur_ksize % 2 == 0 else 0))}")
            blur_ksize = max(3, blur_ksize + (1 if blur_ksize % 2 == 0 else 0))

        # Convertir BGR a LAB
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(img_lab)

        # Estimar el fondo/iluminación en el canal L usando MedianBlur
        # Aplicamos el blur al canal L original
        l_background = cv2.medianBlur(l_channel, blur_ksize)

        # Normalizar el canal L dividiendo por el fondo estimado
        # Convertir a float para la división, añadir epsilon
        l_channel_float = l_channel.astype(np.float32) + 1e-6
        l_background_float = l_background.astype(np.float32) + 1e-6

        # Calcular el canal L normalizado
        mean_l_bg = cv2.mean(l_background)[0] # Usar la media del fondo para escalar
        l_normalized_float = cv2.divide(l_channel_float, l_background_float) * mean_l_bg

        # Recortar valores al rango [0, 255] y convertir a uint8
        l_normalized = np.clip(l_normalized_float, 0, 255).astype(np.uint8)

        # Unir el canal L normalizado con los canales A y B originales
        img_lab_normalized = cv2.merge((l_normalized, a_channel, b_channel))

        # Convertir de nuevo a BGR
        img_bgr_normalized = cv2.cvtColor(img_lab_normalized, cv2.COLOR_LAB2BGR)

        return img_bgr_normalized

    except cv2.error as e:
        print(f"Error de OpenCV en remove_shadows_medianblur_lab_color: {e}")
        return img_bgr # Devolver original en caso de error
    except Exception as e:
        print(f"Error en remove_shadows_medianblur_lab_color: {e}")
        return img_bgr # Devolver original en caso de error
    
    
def remove_shadows_morph_gray(img_bgr, kernel_size=21):
    """
    Removes shadows using morphological closing on grayscale.
    Good for documents or scenes with relatively uniform backgrounds.
    Kernel size might need tuning.
    """
    try:
        # Convert to Grayscale
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Estimate background using morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        # Increase iterations for stronger effect if needed
        background = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Calculate difference (Original - Background)
        # Add a small value to avoid division by zero later if using division
        # We use absolute difference here for simplicity, better methods exist.
        # foreground = cv2.absdiff(img_gray, background)
        # foreground_inv = cv2.bitwise_not(foreground) # Invert to get shadow areas brighter

        # Alternative: Divide original by background (often better for illumination)
        # Ensure background has no zeros and is float
        background_float = background.astype(np.float32) + 1e-6 # Add epsilon
        img_gray_float = img_gray.astype(np.float32)
        normalized_gray = cv2.divide(img_gray_float, background_float) * 255
        normalized_gray = np.clip(normalized_gray, 0, 255).astype(np.uint8) # Clip and convert back to uint8

        # Simple approach: treat the normalized grayscale as the result
        # For color: could apply the normalization factor to color channels, more complex.
        img_bgr_eq = cv2.cvtColor(normalized_gray, cv2.COLOR_GRAY2BGR) # Convert back to BGR
        return img_bgr_eq
    except cv2.error as e:
        print(f"OpenCV Error in remove_shadows_morph_gray: {e}")
        return img_bgr # Return original on error


def process_image_hsv(img):
    """Aplica el procesamiento basado en HSV para eliminar sombras."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Desenfoque gaussiano para reducir ruido y suavizar detalles
    img_blur = cv2.GaussianBlur(img_rgb, (5, 5), 0)
    lab = cv2.cvtColor(img_blur, cv2.COLOR_RGB2LAB)
    # Extraer canal de luminosidad
    L_channel = lab[:, :, 0].astype(float)
    #  Remodela el canal L en una lista de valores de píxeles para aplicar K-Means
    pixel_values = L_channel.reshape((-1, 1)).astype(np.float32)
    #  criterios de terminación para K-Means (máximo de 100 iteraciones o un cambio menor a 0.2).
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    kmeans_mask = labels.reshape(L_channel.shape).astype(np.uint8)
    shadow_cluster = np.argmin(centers)
    shadow_mask_kmeans = (kmeans_mask == shadow_cluster).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    shadow_mask_kmeans_clean = cv2.morphologyEx(shadow_mask_kmeans, cv2.MORPH_OPEN, kernel, iterations=2)

    # Convertir a HSV
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(float)
    h, s, v = cv2.split(hsv)

    # Calcular el valor promedio en la región iluminada (fuera de la sombra)
    illuminated_mask = cv2.bitwise_not(shadow_mask_kmeans_clean)
    illuminated_v_mean = np.mean(v[illuminated_mask > 0])

    # Ajustar el valor en la región de sombra
    v_corrected = v.copy()
    shadow_indices = np.where(shadow_mask_kmeans_clean == 1)
    if illuminated_v_mean > 0 and shadow_indices[0].size > 0:
        gain = illuminated_v_mean / np.mean(v[shadow_indices]) if np.mean(v[shadow_indices]) > 0 else 1.0
        v_corrected[shadow_indices] = np.clip(v[shadow_indices] * gain * 1.3, 0, 255)

    # Recombinar HSV y convertir a RGB
    hsv_corrected = cv2.merge((h, s, v_corrected)).astype(np.uint8)
    corrected_chromaticity = cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2BGR)

    return corrected_chromaticity


def shadow_removal(img_bgr):
    """
    Algoritmo robusto para la eliminación de sombras con ajustes en la reflectancia,
    suavizado y aplicación de máscara de nitidez.
    """
    # 1. Convertir a espacio de color HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    # 2. Estimación de la componente de iluminación
    illumination = cv2.GaussianBlur(v, (51, 51), 0)

    # 3. Estimación de la reflectancia (rango ajustado)
    reflectance = np.clip(v / (illumination + 1e-6), 0, 1.5)

    # 4. Normalización de la reflectancia
    reflectance_normalized = cv2.normalize(reflectance, None, 0, 1, cv2.NORM_MINMAX)

    # 5. Reconstrucción del canal de valor sin sombras
    shadow_free_v = np.clip(reflectance_normalized * 255, 0, 255)

    # 6. Suavizado para reducir artefactos (parámetros ajustados)
    shadow_free_v_smoothed = cv2.bilateralFilter(shadow_free_v.astype(np.uint8), 9, 50, 50).astype(np.float32)

    # 7. Recombinar los canales HSV
    hsv_corrected = cv2.merge((h, s, shadow_free_v_smoothed)).astype(np.uint8)
    shadow_removed_hsv = cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2BGR)

    return shadow_removed_hsv

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Aplica una máscara de nitidez a la imagen."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened

def load_image(image_path):
    """Función para cargar la imagen."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen en la ruta: {image_path}. Verifica la ruta y el archivo.")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

