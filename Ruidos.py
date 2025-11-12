import numpy as np
import skimage as ski
from skimage.color import rgb2gray
from skimage import io
from skimage import metrics
import os
from scipy.ndimage import gaussian_filter
import pandas as pd

DIR_IMGS = "imgs"
DIR_RESULTADOS = "Resultados/resultados_analise" 
TAMANHO_KERNEL = 3
SIGMA_GAUSSIANO = 1.6

# Ruído Gaussiano: sigma=10 -> var=100. Normalizada para [0, 1].
VAR_GAUSSIANO_NORMALIZADA = 100 / (255.0**2) 
# Ruído Sal e Pimenta: 5% de pixels afetados
AMOUNT_SP_5 = 0.05 


def salvar_imagem_uint8(caminho_arquivo, imagem_float_0_1):

    imagem_uint8 = (np.clip(imagem_float_0_1, 0, 1.0) * 255).astype(np.uint8)
    io.imsave(caminho_arquivo, imagem_uint8)

def calcular_metricas(original, filtrada):
    mse = metrics.mean_squared_error(original, filtrada)
    psnr = metrics.peak_signal_noise_ratio(original, filtrada, data_range=1.0)
    return mse, psnr


# Os filtros Média e Mediana são adaptados para uint8 e normalizados de volta.
def filtro_media(img):
    img_uint8 = (img * 255).astype(np.uint8)
    # Filtro da Média (Mean)
    resultado_uint8 = ski.filters.rank.mean(img_uint8, footprint=ski.morphology.square(TAMANHO_KERNEL))
    return resultado_uint8.astype(np.float64) / 255.0

def filtro_mediana(img):
    img_uint8 = (img * 255).astype(np.uint8)
    # Filtro da Mediana
    resultado_uint8 = ski.filters.median(img_uint8, footprint=ski.morphology.square(TAMANHO_KERNEL))
    return resultado_uint8.astype(np.float64) / 255.0

def filtro_gaussiano_scipy(img):
    # O filtro Gaussiano do scipy funciona bem em float[0, 1]
    return gaussian_filter(img, sigma=SIGMA_GAUSSIANO)

FILTROS = {
    "Media": filtro_media, 
    "Mediana": filtro_mediana, 
    "Gaussiano": filtro_gaussiano_scipy
}

# --- Função Principal de Análise por Imagem ---

def realizar_analise_imagem(caminho_completo, nome_arquivo):
    """Carrega, adiciona ruído, filtra, calcula métricas e salva arquivos para uma única imagem."""
    
    nome_base = os.path.splitext(nome_arquivo)[0]
    
    try:
        # 1. Carregamento e Normalização (0-1, float64)
        original_rgb = ski.io.imread(caminho_completo)
        if original_rgb.ndim == 3:
            if original_rgb.shape[-1] == 4:
                original_rgb = original_rgb[:, :, :3]
            original = rgb2gray(original_rgb).astype(np.float64) 
        elif original_rgb.ndim == 2:
             # Trata imagem já em tons de cinza
             original = original_rgb.astype(np.float64) / np.max(original_rgb)
        else:
             print(f"  [AVISO] Imagem '{nome_arquivo}' não é RGB nem Escala de Cinza, ignorando.")
             return []
    
    except Exception as e:
        print(f"  [ERRO] Falha ao carregar ou converter '{nome_arquivo}': {e}")
        return []
    
    print(f"  -> Processando: {nome_arquivo}")

    # Salva a imagem original
    salvar_imagem_uint8(os.path.join(DIR_RESULTADOS, f"{nome_base}_original.png"), original)

    # 2. Adição de Ruído (e.i)
    imagens_ruidosas = {
        "Gaussiano": ski.util.random_noise(original, mode='gaussian', var=VAR_GAUSSIANO_NORMALIZADA),
        "SalEPimenta": ski.util.random_noise(original, mode='s&p', amount=AMOUNT_SP_5, salt_vs_pepper=0.5)
    }

    resultados_imagem = []

    for nome_ruido, img_ruidosa in imagens_ruidosas.items():
        # Salva a imagem ruidosa: nomeArquivo_ruido.png
        nome_arq_ruido = f"{nome_base}_{nome_ruido}.png"
        salvar_imagem_uint8(os.path.join(DIR_RESULTADOS, nome_arq_ruido), img_ruidosa)

        # Métrica da Imagem Ruidosa (base de comparação)
        mse_ruido, psnr_ruido = calcular_metricas(original, img_ruidosa)
        resultados_imagem.append({
            "Arquivo": nome_base,
            "Ruído": nome_ruido,
            "Filtro": "N/A (Ruído)",
            "MSE": f"{mse_ruido:.6f}",
            "PSNR (dB)": f"{psnr_ruido:.2f}"
        })
        
        # 3. Aplicação dos Filtros
        for nome_filtro, filter_func in FILTROS.items():
            try:
                img_filtrada = filter_func(img_ruidosa)
                img_filtrada = np.clip(img_filtrada, 0, 1.0)
                
                # Cálculo de Métricas
                mse, psnr = calcular_metricas(original, img_filtrada)

                nome_arq_filtrado = f"{nome_base}_{nome_ruido}_{nome_filtro}.png"
                salvar_imagem_uint8(os.path.join(DIR_RESULTADOS, nome_arq_filtrado), img_filtrada)
                
                # Armazena métricas
                resultados_imagem.append({
                    "Arquivo": nome_base,
                    "Ruído": nome_ruido,
                    "Filtro": nome_filtro,
                    "MSE": f"{mse:.6f}",
                    "PSNR (dB)": f"{psnr:.2f}"
                })

            except Exception as e:
                print(f"  [ERRO] Falha no filtro {nome_filtro} para ruído {nome_ruido}: {e}")
                resultados_imagem.append({
                    "Arquivo": nome_base,
                    "Ruído": nome_ruido,
                    "Filtro": nome_filtro,
                    "MSE": "ERRO",
                    "PSNR (dB)": "ERRO"
                })
                
    return resultados_imagem


# --- Execução Principal ---
if __name__ == "__main__":
    os.makedirs(DIR_RESULTADOS, exist_ok=True)
    print(f"Os resultados serão salvos no diretório: '{DIR_RESULTADOS}'")

    arquivos_imagem = [f for f in os.listdir(DIR_IMGS) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

    if not arquivos_imagem:
        print(f"Erro: Nenhuma imagem encontrada no diretório '{DIR_IMGS}'. Certifique-se de ter colocado suas imagens lá.")
    else:
        todas_metricas = []
        
        for nome_arquivo in arquivos_imagem:
            caminho_completo = os.path.join(DIR_IMGS, nome_arquivo)
            metricas_atuais = realizar_analise_imagem(caminho_completo, nome_arquivo)
            todas_metricas.extend(metricas_atuais)

        if todas_metricas:
            df_resultados = pd.DataFrame(todas_metricas)
            caminho_tabela = os.path.join(DIR_RESULTADOS, "tabela_metricas_analise.csv")

            df_resultados.to_csv(caminho_tabela, index=False)

            print("\n" + "="*80)
            print(f"Análise concluída para {len(arquivos_imagem)} imagem(s).")
            print(f"As imagens processadas e a tabela de métricas foram salvas em '{DIR_RESULTADOS}'.")
            print("\nConteúdo da Tabela Final (CSV):")
            print("="*80)
            print(df_resultados.to_string(index=False)) # Imprime a tabela no console
            print("="*80)
        else:
            print("\nNenhum resultado de métrica gerado.")