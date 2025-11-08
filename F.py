import cv2
import numpy as np
import skimage as ski
from skimage.color import rgb2gray
from skimage import io
import os
import pandas as pd


def calcular_contraste_local(img_gray):
    """Calcula o Desvio-Padrão Global como proxy para Contraste Local."""
    return np.std(img_gray)

def calcular_nitidez(img_gray):
    """Calcula a Variância do Laplaciano como métrica de Nitidez."""
    laplaciano = cv2.Laplacian(img_gray, cv2.CV_64F)
    return laplaciano.var()


FOTO_ARQUIVO = "file.png"
DOC_ARQUIVO = "documento_texto.png"
INPUT_DIR = "imgs" 

KERNEL_SIZES = [3, 5, 9]
RESULTADOS_DIR = "Resultados/metrica_mini_estudo"

if not os.path.exists(RESULTADOS_DIR):
    os.makedirs(RESULTADOS_DIR)


def realizar_mini_estudo(filename, tipo_imagem):
    filepath = os.path.join(INPUT_DIR, filename)
    
    try:

        img_original = io.imread(filepath)

        if img_original.ndim == 3:

            if img_original.shape[-1] == 4:
                img_rgb = img_original[:, :, :3]
            else:
                img_rgb = img_original
            img_gray = (rgb2gray(img_rgb) * 255).astype(np.uint8)
            
        else:

            img_gray = img_original
    except FileNotFoundError:
        print(f"Erro: O arquivo '{filename}' não foi encontrado. Pulando.")
        return None

    print(f"\nProcessando {tipo_imagem}: {filename} ---")
    
    dados = {
        'Kernel': [], 
        'Contraste (Desv. Padrão)': [], 
        'Nitidez (Var. Laplaciano)': []
    }

    contraste_orig = calcular_contraste_local(img_gray)
    nitidez_orig = calcular_nitidez(img_gray)
    
    dados['Kernel'].append('Original')
    dados['Contraste (Desv. Padrão)'].append(f"{contraste_orig:.2f}")
    dados['Nitidez (Var. Laplaciano)'].append(f"{nitidez_orig:.2f}")
    
    print(f"Original: Contraste={contraste_orig:.2f}, Nitidez={nitidez_orig:.2f}")

    for k in KERNEL_SIZES:
        
        ksize = (k, k)
        img_suavizada = cv2.blur(img_gray, ksize)
        
        contraste_filtrado = calcular_contraste_local(img_suavizada)
        nitidez_filtrada = calcular_nitidez(img_suavizada)
        
        dados['Kernel'].append(f'Média {k}x{k}')
        dados['Contraste (Desv. Padrão)'].append(f"{contraste_filtrado:.2f}")
        dados['Nitidez (Var. Laplaciano)'].append(f"{nitidez_filtrada:.2f}")
        
        print(f"Média {k}x{k}: Contraste={contraste_filtrado:.2f}, Nitidez={nitidez_filtrada:.2f}")

    df_resultado = pd.DataFrame(dados)

    tipo_imagem_seguro = tipo_imagem.replace(' ', '_').replace('/', '-')
    output_filename = os.path.join(RESULTADOS_DIR, f"tabela_mini_estudo_{tipo_imagem_seguro}.txt")
    
    with open(output_filename, 'w') as f:
        f.write(f"Resultados para {tipo_imagem} ({filename}):\n\n")
        f.write(df_resultado.to_string(index=False))
        f.write("\n\n")
        
    print(f"\nTabela salva em: {output_filename}")
    return df_resultado

if __name__ == "__main__":

    df_foto = realizar_mini_estudo(FOTO_ARQUIVO, "Foto (Rosto/Cena)")

    df_doc = realizar_mini_estudo(DOC_ARQUIVO, "Documento (Texto)")
    
    print("\n\n Mini-Estudo Concluído. ")