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

            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), sharex=True, sharey=True)
            ax = axes.ravel()

            fig.suptitle(f"Detecção de Bordas - {filename}", fontsize=16)

            ax[0].imshow(img_original)
            ax[0].set_title("Original")

            ax[1].imshow(img_gray, cmap=plt.cm.gray)
            ax[1].set_title("Escala de Cinza")

            ax[2].imshow(borda_sobel, cmap=plt.cm.gray)
            ax[2].set_title("Sobel (Gradiente)")

            ax[3].imshow(borda_prewitt, cmap=plt.cm.gray)
            ax[3].set_title("Prewitt (Gradiente)")

            ax[4].imshow(borda_sobel_otsu, cmap=plt.cm.gray)
            ax[4].set_title("Sobel + Otsu (Binarizado)")

            ax[5].imshow(borda_canny, cmap=plt.cm.gray)
            ax[5].set_title("Canny (Referência)")

            for a in ax:
                a.axis('off')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            os.makedirs('Resultados/bordas', exist_ok=True)
            plt.savefig(f'Resultados//bordas/borda_{filename}')
            plt.show()

        except Exception as e:
            print(f"Erro ao processar {filename}: {e}")
            
print("Processamento concluído.")