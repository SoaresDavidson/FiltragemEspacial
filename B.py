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




diretorio_imgs = "Resultados\\filtros\\suavizacao"
kernel_sizes = [3, 5, 7]
sigma_values = [0.8, 1.6]
MAX_AMF_KSIZE = 9
FIG_WIDTH = 15
FIG_HEIGHT_PER_ROW = 5



os.makedirs(diretorio_imgs, exist_ok=True)

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
                cv2.imwrite(f'resultados\\filtros\\suavizacao\\Media_{k}x{k}.png', blurred)
                groups["Media"].append((f"Media_{k}x{k}", blurred))

            for s in sigma_values:
                blurred = cv2.GaussianBlur(img_proc, (0, 0), s)
                s_str = str(s).replace('.', '')
                cv2.imwrite(f'resultados/filtros/suavizacao/Gaussiano_S{s_str}.png', blurred)
                groups["Gaussiano"].append((f"Gaussiano_S{s_str}", blurred))

            for k in kernel_sizes:
                blurred = cv2.medianBlur(img_proc, k)
                cv2.imwrite(f'resultados\\filtros\\suavizacao\\Mediana_{k}x{k}.png', blurred)
                groups["Mediana_e_Adaptativo"].append((f"Mediana_{k}x{k}", blurred))

            print(f" -> Aplicando Mediana Adaptativa (Max K={MAX_AMF_KSIZE})...")
            img_MA = mediana_adaptativa(img_proc, max_ksize=MAX_AMF_KSIZE)
            groups["Mediana_e_Adaptativo"].append((f"Mediana_Adaptativa_Max{MAX_AMF_KSIZE}", img_MA))
            
            # Salva cada imagem individualmente com cv2.imwrite
            for group_name, results in groups.items():
                for title, img_out in results:
                    output_filename = f"{base_name}_{title}.jpg"
                    output_filepath = os.path.join(diretorio_imgs, output_filename)
                    print(output_filename)
                    cv2.imwrite(output_filepath, img_out)
                    print(output_filepath)
                    print(f" -> Salvo: {output_filename}")

        except Exception as e:
            print(f"Erro ao processar {filename}: {e}")
            
print("\nDeu certo.")