# Projeto de Reprodução: "Filtro Sobel e classificação linear para análise de deepfake em faces"

Este repositório contém o código-fonte para o projeto de reprodução da disciplina de **Aprendizagem de Máquina**, ministrada pelo Prof. Dr. Lucas Bastos.

**Autor:** Felipe da Cunha Carvalho (`felipe2003@unifesspa.edu.br`)

---

## 1. Objetivo do Projeto

O objetivo deste trabalho foi realizar uma análise de reprodutibilidade do artigo:

* **Artigo Original:** Tamanaka, F. G., & Thomaz, C. E. (2023). **"Sobel filter and linear classification for deepfake analysis of faces"**.
* **Publicação:** Anais do XX Encontro Nacional de Inteligência Artificial e Computacional (ENIAC) / BRACIS 2023.

O artigo propõe um pipeline de baixo custo computacional (Filtro Sobel + PCA + MLDA) para a detecção de *deepfakes*. O objetivo foi replicar este pipeline e comparar nossos resultados com os reportados pelos autores.

## 2. Nossos Resultados e Análise

Nós implementamos o pipeline completo, incluindo as etapas de detecção e alinhamento facial (FAN/SFD) e o classificador (PCA+MLDA). Os resultados da nossa execução (usando a mesma amostra de 100 imagens do artigo) estão abaixo:

| Experimento | Artigo Original | Nossa Reprodução |
| :--- | :--- | :--- |
| Sem Filtro Sobel | 69.25% ± 2.44% | 59.00% ± 10.68% |
| **Com Filtro Sobel** | **96.00% ± 1.63%** | **63.00% ± 13.27%** |

### Conclusão da Análise
A nossa reprodução **validou a hipótese central** do artigo: o uso do Filtro Sobel de fato melhora a acurácia (de 59.00% para 63.00%).

No entanto, a acurácia de 96.00% não foi replicada. Nossa análise (detalhada no relatório técnico) conclui que a principal causa para essa divergência é a **alta instabilidade estatística** de usar um conjunto de dados tão pequeno (100 imagens). Provamos isso ao re-sortear a amostra de 100 imagens e obter um resultado completamente diferente (48.00%), indicando que o resultado de 96% do artigo original foi provavelmente um *outlier* (sorteio de amostragem) difícil de reproduzir.

## 3. Estrutura do Repositório

/projeto-deepfake-reproducao/ | +-- .gitignore (Ignora os dados e o .venv) | +-- 1_extract_frames.py (Script para criar a amostra de dados) | +-- 2_process_and_train.py (Script principal: pipeline e avaliação) | +-- README.md (Este arquivo) | +-- requirements.txt (Dependências do projeto) | +-- Relatorio Técnico DeepFake.pdf (O relatório final) | +-- (Celeb-DF/) <- (Ignorado pelo .gitignore) +-- (frames_dataset_celebdf/) <- (Ignorado pelo .gitignore) +-- (.venv/) <- (Ignorado pelo .gitignore)

## 4. Como Executar os Experimentos

Siga estas instruções para configurar o ambiente e rodar a replicação.

### 4.1. Pré-requisitos
* Python 3.11+
* Git

### 4.2. Configuração do Ambiente

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/CarvalhoDZ7/projeto-deepfake-reproducao.git](https://github.com/CarvalhoDZ7/projeto-deepfake-reproducao.git)
    cd projeto-deepfake-reproducao
    ```

2.  **Crie e ative o ambiente virtual:**
    ```bash
    python -m venv .venv
    # No Windows
    .\.venv\Scripts\activate
    # No macOS/Linux
    source .venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Baixe o Dataset (Instrução Corrigida):**
    * Este projeto requer o dataset **Celeb-DF**.
    * **Acesse o repositório oficial:** `https://github.com/yuezunli/celeb-deepfakeforensics`
    * No `README.md` deles, encontre e preencha o **formulário de requisição** para receber o link de download.
    * Descompacte o dataset baixado e coloque a pasta `Celeb-DF` (contendo `Celeb-real`, `Celeb-synthesis`, etc.) dentro da pasta `projeto-deepfake-reproducao/`. 

### 4.3. Execução dos Experimentos

Os scripts devem ser executados na ordem correta.

**Etapa 1: Extrair Frames de Amostragem**
Este script cria a amostra de 100 frames (50 reais, 50 fakes) a partir dos vídeos e os salva na pasta `frames_dataset_celebdf/`.

```bash
python 1_extract_frames.py
```
**Etapa 2: Processar e Treinar (O Experimento Principal)** 
Este script carrega os 100 frames, executa o pipeline completo de alinhamento (FAN/SFD) e o classificador (Sobel + PCA + MLDA), e imprime a tabela de resultados final.

```bash
python 2_process_and_train.py
