import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Configuração da página do Streamlit
st.set_page_config(page_title="Detector de Imagens - Visão Computacional", layout="centered")

st.title("📸 Identificação Inteligente de Imagens")
st.write("Faça o upload de uma imagem para detectar objetos em tempo real.")

# Inicializa o modelo YOLOv8 Nano (carrega uma vez e guarda em cache)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Botão de upload de arquivos
uploaded_file = st.file_uploader("Selecione uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Converter o arquivo enviado para o formato PIL Image
    image = Image.open(uploaded_file)
    
    # Criar colunas para organizar o ANTES e DEPOIS
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagem Original")
        # Atualizado: use_container_width substitui o parâmetro antigo para remover o aviso amarelo
        st.image(image, use_container_width=True)
        
    # Converter imagem PIL para formato OpenCV (BGR) para o modelo processar
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Executar a inferência/detecção do YOLO
    with st.spinner("Processando e identificando objetos..."):
        results = model(img_bgr)
        
        # O método .plot() renderiza as caixas de detecção automaticamente na imagem original (em BGR)
        annotated_frame = results[0].plot()
        
        # Converter de volta para RGB para exibir corretamente no Streamlit
        img_rgb_result = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(img_rgb_result)
        
    with col2:
        st.subheader("Objetos Identificados")
        # Atualizado: use_container_width aplicado aqui também
        st.image(result_image, use_container_width=True)
        
    # Exibir resumo das detecções em texto abaixo das imagens
    st.success("Processamento concluído com sucesso!")
