import skimage as ski

from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import os

filename = os.path.join("imgs", 'file.png')
original = ski.io.imread(filename)
original = rgb2gray(original)

marcelo_gaussiano = ski.util.random_noise(original, mode='gaussian', var = 10)
marcelo_sp = ski.util.random_noise(original, mode='s&p', amount=0.1, salt_vs_pepper=0.1)

ims_ruidos = [marcelo_gaussiano, marcelo_sp]
filters = [ski.filters.rank.mean, ski.filters.median, ski.filters.gaussian]
filtragens = [original, marcelo_gaussiano, marcelo_sp]

for img in ims_ruidos:
    for filter in filters:
        try:
            filtragens.append(filter(img, footprint=ski.morphology.square(3)))
        except:
            filtragens.append(filter(img, sigma=1))
# Plotar a imagem
fig, axes = plt.subplots(3, 3, figsize=(20, 8))

titles = ["Marcelo","Marcelo - Ruído (Gaussiano)","Marcelo - Ruído (Sal e Pimenta)", 'Média - Gaussian', 'Mediana - Gaussian', 'Gaussiano - Gaussian',
          'Média - Sal e Pimenta', 'Mediana - Sal e Pimenta', 'Gaussiano - Sal e Pimenta']

for i in range(3):
    for j in range(3):
        axes[i, j].imshow(filtragens[(i)*3 + (j)], cmap='gray')
        axes[i, j].set_title(titles[(i)*3 + (j)])
        axes[i, j].axis('off')  # Remove os eixos


plt.show()