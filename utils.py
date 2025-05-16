import os
import matplotlib.pyplot as plt
import cv2
# --- Configuration ---
def get_images(dir):
    res_dir = f'res/{dir}'
    image_files = []
    if os.path.isdir(res_dir): # Verificar si el directorio existe
        # os.listdir() obtiene todos los archivos y carpetas dentro de res_dir
        # Filtramos para quedarnos solo con archivos (no subdirectorios)
        # y opcionalmente, solo con extensiones de imagen comunes
        allowed_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')
        for filename in os.listdir(res_dir):
            file_path = os.path.join(res_dir, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(allowed_extensions):
                image_files.append(filename) # Añadir solo el nombre del archivo a la lista
        print(f"Archivos encontrados en '{res_dir}': {len(image_files)}")
        # print(image_files) # Descomenta si quieres ver la lista completa de archivos
    else:
        print(f"Error: El directorio '{res_dir}' no fue encontrado.")
        image_files = [] # Dejar la lista vacía si el directorio no existe
    images = {img: cv2.imread(f'{res_dir}/{img}')for img in image_files} 
    return images

def get_jd_images():
    return get_images('jhondeer')

def get_moving_case_images():
    return get_images('case_1')

def get_static_case_images():
    return get_images('case_2')

# --- Display function ---
def display_results(processing_function, images_dict_bgr, **kwargs):
    """
    Displays original and processed images side-by-side.

    Args:
        processing_function: A function that takes a BGR image and returns a BGR image.
        images_dict_bgr: A dictionary {filename: bgr_image}.
        **kwargs: Additional keyword arguments passed to the processing_function.
    """
    num_images = len(images_dict_bgr)
    plt.figure(figsize=(12, num_images * 5)) # Adjust figsize as needed
    plot_index = 1

    print(f"--- Applying method: {processing_function.__name__} ---")

    for filename, img_bgr in images_dict_bgr.items():
        if img_bgr is None:
            print(f"Skipping {filename}: Image data is None.")
            continue


        # 2. Apply the processing function
        processed_bgr = processing_function(img_bgr, **kwargs)

        # 3. Convert processed BGR back to RGB (for matplotlib display)
        processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)

        # --- Plotting ---
        # Original Image
        plt.subplot(num_images, 2, plot_index)
        plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        plt.title(f'Original: {filename}')
        plt.axis('off')
        plot_index += 1

        # Processed Image
        plt.subplot(num_images, 2, plot_index)
        plt.imshow(processed_rgb)
        plt.title(f'Processed: {processing_function.__name__}')
        plt.axis('off')
        plot_index += 1

    plt.tight_layout()
    plt.show()

def display_results_two_methods(method1, method2, images_dict_bgr, **kwargs):
    """
    Muestra imágenes originales y procesadas por dos métodos, lado a lado.

    Args:
        method1: Primer método de procesamiento.
        method2: Segundo método de procesamiento.
        images_dict_bgr: Diccionario {filename: bgr_image}.
        **kwargs: Argumentos adicionales para los métodos.
    """
    num_images = len(images_dict_bgr)
    plt.figure(figsize=(18, num_images * 5))
    plot_index = 1

    for filename, img_bgr in images_dict_bgr.items():
        if img_bgr is None:
            print(f"Skipping {filename}: Image data is None.")
            continue

        # Procesar con ambos métodos
        processed1 = method1(img_bgr, **kwargs)
        processed2 = method2(img_bgr, **kwargs)

        # Mostrar original
        plt.subplot(num_images, 3, plot_index)
        plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        plt.title(f'Original: {filename}')
        plt.axis('off')
        plot_index += 1

        # Mostrar método 1
        plt.subplot(num_images, 3, plot_index)
        plt.imshow(cv2.cvtColor(processed1, cv2.COLOR_BGR2RGB))
        plt.title(f'{method1.__name__}')
        plt.axis('off')
        plot_index += 1

        # Mostrar método 2
        plt.subplot(num_images, 3, plot_index)
        plt.imshow(cv2.cvtColor(processed2, cv2.COLOR_BGR2RGB))
        plt.title(f'{method2.__name__}')
        plt.axis('off')
        plot_index += 1

    plt.tight_layout()
    plt.show()