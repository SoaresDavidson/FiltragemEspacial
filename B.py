import cv2
import numpy as np
import skimage as ski
from skimage import io
from matplotlib import pyplot as plt
import os
import math


def mediana_adaptativa(img, max_ksize=9, min_ksize=3):
    H, W = img.shape
    imagem_final = img.copy()
    
    imagem_preenchida = cv2.copyMakeBorder(img, max_ksize//2, max_ksize//2, max_ksize//2, max_ksize//2, cv2.BORDER_REPLICATE)
    for linha in range(H):
        for coluna in range(W):
            tamanho_janela = min_ksize
            while tamanho_janela <= max_ksize:
                inicio_linha = linha + (max_ksize - tamanho_janela) // 2
                fim_linha = inicio_linha + tamanho_janela
                inicio_coluna = coluna + (max_ksize - tamanho_janela) // 2
                fim_coluna = inicio_coluna + tamanho_janela
                janela = imagem_preenchida[inicio_linha:fim_linha, inicio_coluna:fim_coluna]
                valor_minimo = np.min(janela)
                valor_maximo = np.max(janela)
                valor_mediana = np.median(janela)
                valor_pixel_central = img[linha, coluna]

                if valor_mediana > valor_minimo and valor_mediana < valor_maximo:
                    if valor_pixel_central > valor_minimo and valor_pixel_central < valor_maximo:
                        imagem_final[linha, coluna] = valor_pixel_central 
                    else:
                        imagem_final[linha, coluna] = valor_mediana
                    break 

                else:
                    tamanho_janela += 2 
                    if tamanho_janela > max_ksize:
                        imagem_final[linha, coluna] = valor_mediana
                    break
    return imagem_final.astype(np.uint8)




diretorio_imgs = "Resultados/suavizacao"
kernel_sizes = [3, 5, 7]
sigma_values = [0.8, 1.6]
MAX_AMF_KSIZE = 9
FIG_WIDTH = 15
FIG_HEIGHT_PER_ROW = 5


if not os.path.exists(diretorio_imgs):
    os.makedirs(diretorio_imgs)
    print(f"Diretório de saída '{diretorio_imgs}' criado.")
else:
    print(f"Diretório de saída '{diretorio_imgs}' já existe.")

for filename in os.listdir("imgs"):
    filepath = os.path.join("imgs", filename)

    if os.path.isfile(filepath):
        print(f"\nProcessando e agrupando resultados para: **{filename}**")
        base_name, ext = os.path.splitext(filename)

        try:
            img_original = cv2.imread(filepath) 
            
            if img_original is None:
                print(f"Aviso: Não foi possível carregar o arquivo {filename}. Pulando.")
                continue

            if img_original.ndim == 3:
                img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
            elif img_original.ndim == 2:
                img_gray = img_original

            img_proc = img_gray

            groups = {
                "Media": [("Original_Cinza", img_proc)],
                "Gaussiano": [("Original_Cinza", img_proc)],
                "Mediana_e_Adaptativo": [("Original_Cinza", img_proc)],
            }

            for k in kernel_sizes:
                blurred = cv2.blur(img_proc, (k, k))
                groups["Media"].append((f"Media_{k}x{k}", blurred))

            for s in sigma_values:
                blurred = cv2.GaussianBlur(img_proc, (0, 0), s)
                s_str = str(s).replace('.', '')
                groups["Gaussiano"].append((f"Gaussiano_S{s_str}", blurred))

            for k in kernel_sizes:
                blurred = cv2.medianBlur(img_proc, k)
                groups["Mediana_e_Adaptativo"].append((f"Mediana_{k}x{k}", blurred))

            print(f" -> Aplicando Mediana Adaptativa (Max K={MAX_AMF_KSIZE})...")
            img_MA = mediana_adaptativa(img_proc, max_ksize=MAX_AMF_KSIZE)
            groups["Mediana_e_Adaptativo"].append((f"Mediana_Adaptativa_Max{MAX_AMF_KSIZE}", img_MA))
            
            for group_name, results in groups.items():
                num_results = len(results)
                
                cols = 3
                rows = int(math.ceil(num_results / cols))
                
                fig, axes = plt.subplots(rows, cols, figsize=(FIG_WIDTH, rows * FIG_HEIGHT_PER_ROW)) 
                fig.suptitle(f"Filtros {group_name.replace('_', ' ')} vs. Original: {filename}", fontsize=16, fontweight='bold')
                
                ax_flat = axes.flatten()
                
                for i, (title, img_out) in enumerate(results):
                    ax = ax_flat[i]
                    ax.imshow(img_out, cmap='gray')
                    ax.set_title(title.replace('_', ' '), fontsize=12)
                    ax.axis('off')

                for j in range(num_results, len(ax_flat)):
                    fig.delaxes(ax_flat[j])

                plt.tight_layout(rect=[0, 0.03, 1, 0.96])
                
                output_filename = f"{base_name}_Grupo_{group_name}.png"
                output_filepath = os.path.join(diretorio_imgs, output_filename)
                
                plt.savefig(output_filepath)
                plt.close(fig) 
                
                print(f" -> Grupo '{group_name}' salvo em: {output_filename}")

        except Exception as e:
            print(f"Erro ao processar {filename}: {e}")
            
print("\nDeu certo.")