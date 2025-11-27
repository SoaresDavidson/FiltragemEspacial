import cv2
import numpy as np
from skimage.morphology import skeletonize
import os

# --- 1. FUNÇÃO DE PRÉ-PROCESSAMENTO ---
def pre_processamento_avancado(img):
    # Normalização
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    # Median Blur (Sujeira)
    img_noise = cv2.medianBlur(img_norm, 3)
    # CLAHE (Contraste)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_noise)
    # Gaussian Blur (Suavização)
    img_final = cv2.GaussianBlur(img_clahe, (5, 5), 0)
    return img_final

# --- 2. PROCESSAMENTO E SALVAMENTO INDIVIDUAL ---
def processar_e_salvar_separado(caminho_entrada, pasta_raiz_saida, nome_arquivo):
    print(f"--> Processando: {nome_arquivo}...")
    
    # Criar uma subpasta exclusiva para essa imagem
    # Ex: se o arquivo é "dedo1.jpg", cria a pasta "resultados/dedo1"
    nome_sem_extensao = os.path.splitext(nome_arquivo)[0]
    pasta_destino = os.path.join(pasta_raiz_saida, nome_sem_extensao)
    
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)

    # 1. Carregar Original
    img_original = cv2.imread(caminho_entrada, 0)
    if img_original is None: return
    
    # Salvar 01
    cv2.imwrite(os.path.join(pasta_destino, "01_original.png"), img_original)

    # 2. Pré-processamento
    img_pre = pre_processamento_avancado(img_original)
    # Salvar 02
    cv2.imwrite(os.path.join(pasta_destino, "02_pre_processada.png"), img_pre)

    # 3. Máscara (Convex Hull)
    _, mask_thresh = cv2.threshold(img_pre, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    mask_closed = cv2.morphologyEx(mask_thresh, cv2.MORPH_CLOSE, kernel_mask)
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask_final = np.zeros_like(img_original)
    if contours:
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        cv2.drawContours(mask_final, [hull], -1, 255, -1)
        mask_final = cv2.dilate(mask_final, kernel_mask, iterations=2)
    
    # Salvar 03 (Opcional, para ver a máscara)
    cv2.imwrite(os.path.join(pasta_destino, "03_mascara_roi.png"), mask_final)

    # 4. Segmentação (C=7)
    img_adapt = cv2.adaptiveThreshold(img_pre, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 
                                      25, 6)
    img_masked = cv2.bitwise_and(img_adapt, img_adapt, mask=mask_final)
    
    # Salvar 04
    cv2.imwrite(os.path.join(pasta_destino, "04_segmentacao_bruta.png"), img_masked)

    # 5. Morfologia (Limpeza)
    kernel_morf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_open = cv2.morphologyEx(img_masked, cv2.MORPH_OPEN, kernel_morf)
    img_closed = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel_morf)
    
    # Salvar 05
    cv2.imwrite(os.path.join(pasta_destino, "05_morfologia_limpa.png"), img_closed)

    # 6. Esqueletização
    skeleton = skeletonize(img_closed > 0)
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)
    
    # Salvar 06
    cv2.imwrite(os.path.join(pasta_destino, "06_esqueleto.png"), skeleton_uint8)

    # 7. Minúcias
    esqueleto_norm = skeleton_uint8 // 255
    kernel_vizinhos = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    vizinhos = cv2.filter2D(esqueleto_norm, -1, kernel_vizinhos) * esqueleto_norm
    
    terminacoes = (vizinhos == 1)
    bifurcacoes = (vizinhos == 3)
    
    mask_minucias = cv2.erode(mask_final, kernel_mask, iterations=3)
    terminacoes = terminacoes & (mask_minucias > 0)
    bifurcacoes = bifurcacoes & (mask_minucias > 0)

    # Desenhar na imagem colorida
    res_visual = cv2.cvtColor(skeleton_uint8, cv2.COLOR_GRAY2BGR)
    y, x = np.where(terminacoes)
    for px, py in zip(x, y): cv2.circle(res_visual, (px, py), 2, (0, 0, 255), 1) # Vermelho
    y, x = np.where(bifurcacoes)
    for px, py in zip(x, y): cv2.circle(res_visual, (px, py), 2, (255, 0, 0), 1) # Azul

    # Salvar 07 (Resultado Final)
    cv2.imwrite(os.path.join(pasta_destino, "07_resultado_final.png"), res_visual)

# --- 3. LOOP PRINCIPAL ---
def processar_lote():
    pasta_entrada = "digitais"
    pasta_saida = "resultados_separados"

    if not os.path.exists(pasta_entrada):
        print(f"Crie a pasta '{pasta_entrada}' e coloque as imagens lá.")
        return
        
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)

    arquivos = os.listdir(pasta_entrada)
    validos = ('.jpg', '.png', '.jpeg', '.tif', '.bmp')

    contador = 0
    for arq in arquivos:
        if arq.lower().endswith(validos):
            processar_e_salvar_separado(os.path.join(pasta_entrada, arq), pasta_saida, arq)
            contador += 1
            
    print(f"\nConcluído! {contador} imagens processadas.")
    print(f"Verifique a pasta '{pasta_saida}'. Cada imagem tem sua própria subpasta.")

if __name__ == "__main__":
    processar_lote()