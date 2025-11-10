import cv2
import os
import random
import numpy as np

# --- 1. CONFIGURE SEUS CAMINHOS AQUI ---

# Coloque o caminho para a sua pasta principal do Celeb-DF
# Se a pasta "Celeb-DF" está no mesmo local que este script,
# deixe como está: "Celeb-DF"
BASE_DATASET_PATH = "Celeb-DF" 

# Estas são as pastas de origem
PATH_CELEB_REAL = os.path.join(BASE_DATASET_PATH, "Celeb-real")
PATH_YOUTUBE_REAL = os.path.join(BASE_DATASET_PATH, "YouTube-real")
PATH_CELEB_FAKE = os.path.join(BASE_DATASET_PATH, "Celeb-synthesis")

# Esta é a pasta de destino para salvar os frames
# Crie esta pasta (vazia) antes de rodar o script
OUTPUT_FOLDER = "frames_dataset_celebdf"

# Número de frames para extrair (50 de cada, como no artigo)
NUM_FRAMES_PER_CLASS = 50

# --- FIM DA CONFIGURAÇÃO ---


def get_all_video_paths(folders):
    """Pega uma lista de todos os arquivos .mp4 dentro de uma lista de pastas."""
    video_paths = []
    for folder in folders:
        if not os.path.exists(folder):
            print(f"AVISO: Pasta não encontrada. Pulando: {folder}")
            print(f"       Verifique se o caminho '{BASE_DATASET_PATH}' está correto.")
            continue
        for filename in os.listdir(folder):
            if filename.endswith(".mp4"):
                video_paths.append(os.path.join(folder, filename))
    return video_paths

def extract_middle_frame(video_path):
    """Abre um vídeo e retorna o frame do meio."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir vídeo: {video_path}")
        return None, False

    # Pega o número total de frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define o frame do meio
    middle_frame_index = total_frames // 2
    
    # Define o leitor para o frame do meio
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Erro ao ler frame do vídeo: {video_path}")
        return None, False
        
    return frame, True

def process_videos(video_list, output_path, label_prefix):
    """
    Processa uma lista de vídeos, extrai o frame do meio e salva na pasta de saída.
    """
    # Garante que a pasta de saída exista
    os.makedirs(output_path, exist_ok=True)
    
    # Embaralha a lista para pegar vídeos aleatórios
    random.shuffle(video_list)
    
    count = 0
    video_index = 0
    
    while count < NUM_FRAMES_PER_CLASS and video_index < len(video_list):
        video_path = video_list[video_index]
        video_index += 1 # Passa para o próximo vídeo
        
        frame, success = extract_middle_frame(video_path)
        
        if success:
            # Salva o frame
            output_filename = f"{label_prefix}_{count}.png"
            output_filepath = os.path.join(output_path, output_filename)
            cv2.imwrite(output_filepath, frame)
            
            count += 1
            print(f"[{count}/{NUM_FRAMES_PER_CLASS}] Salvo: {output_filepath}")
        else:
            print(f"Falha ao processar: {video_path}. Tentando próximo vídeo.")

    print(f"\nProcessamento de '{label_prefix}' concluído. Total salvo: {count}")

# --- EXECUÇÃO DO SCRIPT ---
if __name__ == "__main__":
    print("--- Iniciando Extração de Frames (Celeb-DF) ---")

    # 1. Processar Vídeos REAIS
    print("\nProcessando vídeos REAIS...")
    real_folders = [PATH_CELEB_REAL, PATH_YOUTUBE_REAL]
    all_real_videos = get_all_video_paths(real_folders)
    if len(all_real_videos) > 0:
        process_videos(all_real_videos, OUTPUT_FOLDER, "real_celeb")
    else:
        print("Nenhum vídeo real encontrado. Verifique seus caminhos.")

    # 2. Processar Vídeos FAKES
    print("\nProcessando vídeos FAKES...")
    fake_folders = [PATH_CELEB_FAKE]
    all_fake_videos = get_all_video_paths(fake_folders)
    if len(all_fake_videos) > 0:
        process_videos(all_fake_videos, OUTPUT_FOLDER, "fake_celeb")
    else:
        print("Nenhum vídeo fake encontrado. Verifique seus caminhos.")

    print("\n--- Extração de Frames Concluída ---")
    print(f"Verifique a pasta '{OUTPUT_FOLDER}' pelas 100 imagens.")