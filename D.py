import cv2
import numpy as np
import skimage as ski
from skimage import filters
from skimage.color import rgb2gray
from skimage import io
from matplotlib import pyplot as plt
import os


for filename in os.listdir("imgs"):
 
    filepath = os.path.join("imgs", filename)

    if os.path.isfile(filepath):
        print(f"Processando imagem: {filename}")
        
        try:
  
            img_original = ski.io.imread(filepath)
            if img_original.ndim == 3:
                img_gray = rgb2gray(img_original)
            elif img_original.ndim == 2:
                img_gray = img_original / 255.0 if img_original.dtype == np.uint8 else img_original
            else:
                print(f"Pulando {filename}: Formato de imagem inesperado.")
                continue

            borda_sobel = filters.sobel(img_gray)

            borda_prewitt = filters.prewitt(img_gray)

            img_blur = cv2.GaussianBlur((img_gray * 255).astype(np.uint8), (5, 5), 0)
            borda_canny = cv2.Canny(img_blur, 50, 150) / 255.0 

            thresh = filters.threshold_otsu(borda_sobel)
            borda_sobel_otsu = borda_sobel > thresh 

            # Salvamento Individual
            os.makedirs('Resultados/filtros/bordas', exist_ok=True)
            base_name = os.path.splitext(filename)[0]
            
            # Converter para uint8 para salvar
            img_gray_uint8 = (img_gray * 255).astype(np.uint8)
            borda_sobel_uint8 = (borda_sobel * 255).astype(np.uint8)
            borda_prewitt_uint8 = (borda_prewitt * 255).astype(np.uint8)
            borda_sobel_otsu_uint8 = (borda_sobel_otsu * 255).astype(np.uint8)
            borda_canny_uint8 = (borda_canny * 255).astype(np.uint8)
            
            if img_original.ndim == 3:
                cv2.imwrite(f'Resultados/bordas/{base_name}_original.png', cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(f'Resultados/bordas/{base_name}_original.png', img_original)
            
            cv2.imwrite(f'Resultados/filtros/bordas/{base_name}_cinza.png', img_gray_uint8)
            cv2.imwrite(f'Resultados/filtros/bordas/{base_name}_sobel.png', borda_sobel_uint8)
            cv2.imwrite(f'Resultados/filtros/bordas/{base_name}_prewitt.png', borda_prewitt_uint8)
            cv2.imwrite(f'Resultados/filtros/bordas/{base_name}_sobel_otsu.png', borda_sobel_otsu_uint8)
            cv2.imwrite(f'Resultados/filtros/bordas/{base_name}_canny.png', borda_canny_uint8)
            
            print(f"Salvo: {base_name} - 6 imagens de detecção de bordas")

        except Exception as e:
            print(f"Erro ao processar {filename}: {e}")
            
print("Processamento concluído.")