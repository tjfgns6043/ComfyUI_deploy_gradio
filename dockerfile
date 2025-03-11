# NGC PyTorch 컨테이너 (CUDA 12.8 지원)
FROM nvcr.io/nvidia/pytorch:25.02-py3

# 작업 디렉토리 설정
WORKDIR /app

# Miniconda 설치
RUN curl -o ~/miniconda.sh -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -ya

# 환경 변수 설정 (Conda 기본 환경 활성화)
ENV PATH="/opt/conda/bin:$PATH"
SHELL ["/bin/bash", "-c"]

# Conda 가상 환경 생성
RUN conda create -n comfy_env

# Conda 환경 활성화 & PyTorch Nightly 설치 (CUDA 12.8 지원)
RUN conda run -n comfy_env pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 추가 패키지 설치 (Gradio, Requests)
RUN conda run -n comfy_env pip install gradio requests

# 프로젝트 파일 복사
COPY ./ /app/

# ComfyUI 의존성 설치
RUN conda run -n comfy_env pip install -r /app/ComfyUI/requirements.txt

# NGROK 설치
RUN wget -O ngrok.tgz https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-stable-linux-amd64.tgz && \
    tar -xvzf ngrok.tgz && \
    mv ngrok /usr/local/bin/ && \
    chmod +x /usr/local/bin/ngrok && \
    rm ngrok.tgz

RUN ngrok config add-authtoken Your_autoToken

# Ollama 설치
RUN curl -fsSL https://ollama.ai/install.sh | sh

# 실행 스크립트 복사 및 실행 권한 부여
COPY start_services.sh /app/start_services.sh
RUN chmod +x /app/start_services.sh

# EXPOSE 포트 설정
EXPOSE 11434
EXPOSE 7860

# 실행 명령어 (Conda 환경에서 실행)
CMD ["bash", "-c", "source activate comfy_env && bash /app/start_services.sh"]
