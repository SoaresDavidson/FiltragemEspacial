#Unsharp masking, High boost, Laplaciano
import skimage as ski

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
    img_high_boost = img + 2.5 * img_nitido


    # Plotar a imagem
    fig, axes = plt.subplots(1, 5, figsize=(20, 8))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f'{filename}')
    axes[0].axis('off')  

    axes[1].imshow(img_nitido, cmap='gray')
    axes[1].set_title(f'{filename} - Nitido (Unsharp Mask)')
    axes[1].axis('off')  

    axes[2].imshow(img_laplace, cmap='gray')
    axes[2].set_title(f'{filename} - Laplaciano')
    axes[2].axis('off')  

    axes[3].imshow(img_laplace_127, cmap='gray', vmin=0.5, vmax=1)
    axes[3].set_title(f'{filename} - Laplaciano + 127')
    axes[3].axis('off')  

    axes[4].imshow(img_high_boost, cmap='gray')
    axes[4].set_title(f'{filename} - High Boost')
    axes[4].axis('off')  

    plt.show()