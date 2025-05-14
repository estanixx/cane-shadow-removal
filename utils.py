import os
import matplotlib.pyplot as plt
import cv2
# --- Configuration ---
res_dir = 'res'
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


images = {img: cv2.imread(f'res/{img}')for img in image_files}

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
