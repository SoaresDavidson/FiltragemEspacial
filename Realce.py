#Unsharp masking, High boost, Laplaciano
import skimage as ski
import cv2
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import os

for filename in os.listdir("imgs"):
    img = os.path.join("imgs", filename)
    img = ski.io.imread(img)
    img = rgb2gray(img)

    import numpy as np
    
    img_nitido = ski.filters.unsharp_mask(img, radius=1, amount=1)
    img_laplace = ski.filters.laplace(img)
    # Adicionar 0.5 e clipar entre 0 e 1 para visualização correta
    img_laplace_127 = np.clip(img_laplace + 0.5, 0, 1)
    img_high_boost = img + 1.5 * img_laplace


    # Salvamento Individual
    os.makedirs('Resultados/filtros/realce', exist_ok=True)
    base_name = os.path.splitext(filename)[0]
    
    # Converter para uint8 para salvar
    img_uint8 = (img * 255).astype(np.uint8)
    img_nitido_uint8 = np.clip(img_nitido * 255, 0, 255).astype(np.uint8)
    img_laplace_uint8 = np.clip((img_laplace + 1) * 127.5, 0, 255).astype(np.uint8)
    img_laplace_127_uint8 = np.clip(img_laplace_127 * 255, 0, 255).astype(np.uint8)
    img_high_boost_uint8 = np.clip(img_high_boost * 255, 0, 255).astype(np.uint8)
    
    cv2.imwrite(f'Resultados/filtros/realce/{base_name}_original.png', img_uint8)
    cv2.imwrite(f'Resultados/filtros/realce/{base_name}_nitido.png', img_nitido_uint8)
    cv2.imwrite(f'Resultados/filtros/realce/{base_name}_laplaciano.png', img_laplace_uint8)
    cv2.imwrite(f'Resultados/filtros/realce/{base_name}_laplaciano_127.png', img_laplace_127_uint8)
    cv2.imwrite(f'Resultados/filtros/realce/{base_name}_high_boost.png', img_high_boost_uint8)
    
    print(f"Salvo: {base_name} - 5 imagens de realce")