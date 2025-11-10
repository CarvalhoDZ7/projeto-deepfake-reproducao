import cv2
import face_alignment # Biblioteca que contém FAN e SFD
import os
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import warnings

# Ignora avisos
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# --- 1. CONFIGURE OS CAMINHOS ---
DATA_FOLDER = "frames_dataset_celebdf" 
IMAGE_SIZE = (224, 224) # Tamanho final da imagem
# --- FIM DA CONFIGURAÇÃO ---

# --- MUDANÇA PRINCIPAL: INICIALIZA O FAN + SFD ---
print("Carregando modelos FAN (alinhador) e SFD (detector)...")
try:
    # device='cuda' usa a GPU se você tiver uma NVIDIA
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda')
    print("Biblioteca rodando na GPU (cuda).")
except Exception:
    # 'cpu' será MUITO mais lento, mas funciona em qualquer PC
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu')
    print("GPU não encontrada. Rodando na CPU (será lento).")
print("Modelos carregados.")


# --- PONTOS-CHAVE PARA O ALINHAMENTO ---
# Esta é a parte que faltava. Vamos alinhar os rostos para que
# os olhos e o nariz fiquem sempre na mesma posição.

# Pontos de origem no modelo de 68 pontos:
# Ponto 36: Canto externo do olho esquerdo
# Ponto 45: Canto externo do olho direito
# Ponto 30: Ponta do nariz
SRC_PTS = np.float32([
    [0, 0], # Será preenchido pelos pontos detectados
    [0, 0],
    [0, 0]
])

# Pontos de destino na nossa imagem final de 224x224
# (Estes valores são uma escolha de design padrão para alinhar rostos)
DST_PTS = np.float32([
    [70, 90],  # Posição final do olho esquerdo
    [154, 90], # Posição final do olho direito
    [112, 140] # Posição final do nariz
])


def aplicar_filtro_sobel(img_cinza):
    """ Aplica o Filtro Sobel para detectar bordas. """
    sobel_x = cv2.Sobel(img_cinza, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_cinza, cv2.CV_64F, 0, 1, ksize=3)
    img_sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    img_sobel = cv2.normalize(img_sobel, None, 0, 255, cv2.NORM_MINMAX)
    return img_sobel.astype(np.uint8)

def processar_frame(filepath):
    """
    MUDANÇA: Usa FAN/SFD para DETECTAR e ALINHAR o rosto.
    Esta é a replicação exata do pré-processamento do artigo.
    """
    img = cv2.imread(filepath)
    if img is None:
        print(f"Erro ao ler: {filepath}")
        return None, None
        
    img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Converte para RGB (a biblioteca espera RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. Detecta e Alinha (FAN + SFD)
    # Pede à biblioteca para encontrar os 68 pontos (landmarks)
    landmarks_list = fa.get_landmarks(img_rgb)
    
    if landmarks_list is None or len(landmarks_list) == 0:
        # Fallback: Se não achar rosto, redimensiona a imagem inteira
        # (Isso é ruim, mas melhor que falhar)
        rosto_alinhado_cinza = cv2.resize(img_cinza, IMAGE_SIZE)
    else:
        # Pega os 68 pontos do primeiro rosto
        landmarks = landmarks_list[0]
        
        # 2. Pega os 3 pontos-chave que definimos
        SRC_PTS[0] = landmarks[36] # Olho esquerdo
        SRC_PTS[1] = landmarks[45] # Olho direito
        SRC_PTS[2] = landmarks[30] # Nariz
        
        # 3. Calcula a Matriz de Transformação
        # Compara os pontos de origem (onde o rosto está)
        # com os pontos de destino (para onde queremos mover)
        M = cv2.getAffineTransform(SRC_PTS, DST_PTS)
        
        # 4. Aplica a transformação (Warp)
        # Gira, estica e move a imagem de tons de cinza
        # para que ela se encaixe nos nossos pontos de destino
        rosto_alinhado_cinza = cv2.warpAffine(img_cinza, M, IMAGE_SIZE)

    # Agora temos um rosto perfeitamente alinhado de 224x224
    
    # Versão 1: Imagem normal (para o "Sem Sobel")
    img_sem_sobel = rosto_alinhado_cinza.flatten()
    
    # Versão 2: Imagem com Sobel
    img_com_sobel = aplicar_filtro_sobel(rosto_alinhado_cinza)
    img_com_sobel = img_com_sobel.flatten()
    
    return img_sem_sobel, img_com_sobel

# --- ETAPA 4.1: PRÉ-PROCESSAMENTO (CARREGAR DADOS) ---
print(f"Carregando e processando frames da pasta: {DATA_FOLDER}")

X_sem_sobel_list = []
X_com_sobel_list = []
y_list = [] 

# Pega o número de arquivos reais e fakes
reais_paths = [os.path.join(DATA_FOLDER, f) for f in os.listdir(DATA_FOLDER) if f.startswith("real_")]
fakes_paths = [os.path.join(DATA_FOLDER, f) for f in os.listdir(DATA_FOLDER) if f.startswith("fake_")]
total_reais = len(reais_paths)
total_fakes = len(fakes_paths)

if total_reais != total_fakes:
    print(f"AVISO: Dataset desbalanceado! {total_reais} reais vs {total_fakes} fakes.")
    print("O resultado da acurácia pode não ser confiável.")
else:
    print(f"Dataset balanceado: {total_reais} reais e {total_fakes} fakes.")

print("Processando imagens (isso pode demorar MUITO)...")
total_imagens = total_reais + total_fakes
img_count = 0

# Carrega Reais
for filepath in reais_paths:
    img_sem, img_com = processar_frame(filepath)
    if img_sem is not None:
        X_sem_sobel_list.append(img_sem)
        X_com_sobel_list.append(img_com)
        y_list.append(0) # 0 = Real
    img_count += 1
    print(f"Processado {img_count}/{total_imagens}...", end='\r')

# Carrega Fakes
for filepath in fakes_paths:
    img_sem, img_com = processar_frame(filepath)
    if img_sem is not None:
        X_sem_sobel_list.append(img_sem)
        X_com_sobel_list.append(img_com)
        y_list.append(1) # 1 = Fake
    img_count += 1
    print(f"Processado {img_count}/{total_imagens}...", end='\r')

# Converte listas para arrays numpy
X_sem_sobel = np.array(X_sem_sobel_list)
X_com_sobel = np.array(X_com_sobel_list)
y = np.array(y_list)

print(f"\nProcessamento concluído. Total de imagens: {len(y)}")
print(f"Shape dos dados (X): {X_com_sobel.shape}") 


# --- ETAPA 4.2 e 4.3: MODELO E AVALIAÇÃO ---
print("\n--- Iniciando Etapa 4.2 e 4.3: Modelo e Avaliação ---")

# Definição do Modelo: Pipeline [PCA + MLDA]
pipeline_artigo = Pipeline([
    ('scaler', StandardScaler()), 
    ('pca', PCA(n_components=0.99)), # Fiel ao artigo ("todas componentes")
    ('mlda', LinearDiscriminantAnalysis())
])

# Configuração da Avaliação: K-fold com k=5
kfold = KFold(n_splits=5, shuffle=True, random_state=42)


# --- ETAPA 5: EXECUÇÃO DOS EXPERIMENTOS ---
print("\n--- Iniciando Etapa 5: Execução dos Experimentos ---")

# 1. Rodar experimento SEM Filtro Sobel
print("Avaliando pipeline SEM Sobel...")
scores_sem_sobel = cross_val_score(pipeline_artigo, X_sem_sobel, y, cv=kfold, scoring='accuracy')

# 2. Rodar experimento COM Filtro Sobel
print("Avaliando pipeline COM Sobel...")
scores_com_sobel = cross_val_score(pipeline_artigo, X_com_sobel, y, cv=kfold, scoring='accuracy')


# --- ETAPA 6: ANÁLISE CRÍTICA (Coleta de Resultados) ---
print("\n--- Resultados (Para sua Tabela de Comparação) ---")
print(f"Dataset: Celeb-DF ({len(y)} frames balanceados)")

print("\n--- RESULTADOS DA SUA REPRODUÇÃO (com ALINHAMENTO FAN/SFD) ---")
print(f"Acurácia Média (Sem Sobel): {np.mean(scores_sem_sobel)*100:.2f}% ± {np.std(scores_sem_sobel)*100:.2f}%")
print(f"Acurácia Média (Com Sobel): {np.mean(scores_com_sobel)*100:.2f}% ± {np.std(scores_com_sobel)*100:.2f}%")

print("\n--- RESULTADOS DO ARTIGO ORIGINAL (Tabela 1) ---")
print(f"Acurácia Média (Sem Sobel): 69.25% ± 2.44%")
print(f"Acurácia Média (Com Sobel): 96.00% ± 1.63%")