import cv2
import numpy as np
import skimage as ski
from skimage.color import rgb2gray
from skimage import io
import matplotlib.pyplot as plt
import os
import time

# --- Funções de Convolução ---

def aplicar_convolucao_manual(imagem_entrada, kernel):

    imagem_float = imagem_entrada.astype(np.float32)
    
    # Obtém dimensões da imagem e do kernel
    (altura_img, largura_img) = imagem_float.shape
    (altura_kernel, largura_kernel) = kernel.shape
    
    # Calcula o padding necessário (assumindo kernels ímpares)
    padding = (largura_kernel - 1) // 2
    
    # Cria a imagem de saída, que terá o mesmo tamanho da original
    imagem_saida = np.zeros((altura_img, largura_img), dtype=np.float32)
    
    # Aplica o preenchimento de zeros (padding) nas bordas
    imagem_com_padding = cv2.copyMakeBorder(imagem_float, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    
    # Loop principal: iteração pixel a pixel
    # Itera sobre as coordenadas (y, x) da imagem de saída
    for y in range(altura_img):
        for x in range(largura_img):
            # Extrai a ROI (Região de Interesse) - a janela sob o kernel
            # A janela é extraída da imagem com padding
            regiao_interesse = imagem_com_padding[y:y + altura_kernel, x:x + largura_kernel]
            
            # Executa a operação de convolução: multiplicação elemento a elemento
            # entre a ROI e o kernel, seguida pela soma.
            valor = (regiao_interesse * kernel).sum()
            
            # Armazena o valor resultante no pixel (y, x) da imagem de saída
            imagem_saida[y, x] = valor
            
    return imagem_saida
for filename in os.listdir('imgs'):
    # --- Configurações ---
    NOME_ARQUIVO_IMAGEM = filename
    DIRETORIO_ENTRADA = "imgs" 
    caminho_completo_imagem = os.path.join(DIRETORIO_ENTRADA, NOME_ARQUIVO_IMAGEM)

    # 1. Definição dos Kernels Manuais 3x3
    # Kernel da Média 3x3: Suavização (todos os pesos iguais, somam 1)
    kernel_media_3x3 = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ], dtype=np.float32) / 9.0

    # Kernel Laplaciano 3x3: Detecção de Bordas / Nitidez (Soma dos elementos é zero)
    kernel_laplaciano_3x3 = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ], dtype=np.float32)

    # --- Processamento ---


    imagem_original = io.imread(caminho_completo_imagem)
    
    # Garantir que a imagem esteja em escala de cinza e UINT8 (0-255)
    if imagem_original.ndim == 3:
        if imagem_original.shape[-1] == 4:
            # Imagem RGBA (4 canais) -> remove o Alpha
            imagem_rgb = imagem_original[:, :, :3]
        else:
            # Imagem RGB (3 canais)
            imagem_rgb = imagem_original
        
        # Converte para escala de cinza
        imagem_cinza = (rgb2gray(imagem_rgb) * 255).astype(np.uint8)
    else:
        imagem_cinza = imagem_original
            


    print(f"\n--- Aplicando Convolução Manual Pura ({imagem_cinza.shape[0]}x{imagem_cinza.shape[1]}) ---")

    tempo_inicio = time.time()

    # A) Convolução Média
    imagem_media_float = aplicar_convolucao_manual(imagem_cinza, kernel_media_3x3)
    imagem_media_uint8 = np.uint8(np.clip(imagem_media_float, 0, 255))

    #Laplaciano
    imagem_laplaciano_float = aplicar_convolucao_manual(imagem_cinza, kernel_laplaciano_3x3)

    tempo_fim = time.time()
    print(f"Tempo de execução (Python Puro): {tempo_fim - tempo_inicio:.4f} segundos")

    # 3. Processamento do Laplaciano para Visualização
    # O Laplaciano tem valores positivos e negativos. Para visualização (UINT8, 0-255), normalizamos.
    imagem_laplaciano_norm = cv2.normalize(imagem_laplaciano_float, None, 0, 255, cv2.NORM_MINMAX)
    imagem_laplaciano_uint8 = np.uint8(imagem_laplaciano_norm)

    # --- Visualização ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax = axes.ravel()

    ax[0].imshow(imagem_cinza, cmap='gray')
    ax[0].set_title("1. Imagem Original (Cinza)")
    ax[0].axis('off')

    ax[1].imshow(imagem_media_uint8, cmap='gray')
    ax[1].set_title(f"2. Média 3x3 (Convolução Manual)")
    ax[1].axis('off')

    ax[2].imshow(imagem_laplaciano_uint8, cmap='gray')
    ax[2].set_title(f"3. Laplaciano 3x3 (Bordas Normalizado)")
    ax[2].axis('off')

    plt.tight_layout()
    os.makedirs('Resultados/convolução_manual', exist_ok=True)
    plt.savefig(f'Resultados//convolução_manual/media_{filename}')
    plt.show()
