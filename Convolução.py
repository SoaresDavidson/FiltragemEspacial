import skimage as ski
from skimage.color import rgb2gray
from skimage.exposure import histogram
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import os

for filename in os.listdir('imgs'):
    img = ski.io.imread(os.path.join('imgs', filename))
    img = rgb2gray(img)
    # Converter para uint8 [0, 255]
    img = img_as_ubyte(img)
    # Aplicar filtro da média
    img_media = ski.filters.rank.mean(img.copy(), ski.morphology.square(3))

    #histograma das imagens
    hist_img, bins_img = histogram(img, source_range='dtype')
    hist_img_media, bins_img_media = histogram(img_media, source_range='dtype')

    # Plotar a imagem
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title(filename)
    axes[0, 0].axis('off')  # Remove os eixos

    axes[0, 1].imshow(img_media, cmap='gray')
    axes[0, 1].set_title(f'{filename} - Média')
    axes[0, 1].axis('off')  # Remove os eixos
    
    axes[1, 0].plot(bins_img, hist_img, color='blue')
    axes[1, 0].set_title(f'Histograma - Original')
    axes[1, 0].set_xlabel('Intensidade do Pixel')
    axes[1, 0].set_ylabel('Contagem de Pixels')
    axes[1, 0].set_xlim([0, 255])
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(bins_img_media, hist_img_media, color='red')
    axes[1, 1].set_title(f'Histograma - Média')
    axes[1, 1].set_xlabel('Intensidade do Pixel')
    axes[1, 1].set_ylabel('Contagem de Pixels')
    axes[1, 1].set_xlim([0, 255])
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()


    os.makedirs('Resultados/convolução', exist_ok=True)
    plt.savefig(f'Resultados//convolução/media_{filename}')
    plt.show()