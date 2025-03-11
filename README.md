# 🚀 Flux Dev AI Image Generation

Flux Dev is a **Stable Diffusion 3-based AI image generation system**. Users can input prompts through the Gradio web UI, which translates Korean prompts into English using Ollama before generating images with Stable Diffusion 3. Additionally, Ngrok is used to enable remote access to the web UI.

## 📌 **Workflow Diagram**
The following diagram illustrates the overall system architecture:

![flux_workflow](https://github.com/user-attachments/assets/70d73ed8-b13a-426b-ba87-1ae8fc66a6a3)

## 🔧 **Installation & Execution**
### 1️⃣ **Running with Docker**
```bash
git clone https://github.com/tjfgns6043/ComfyUI_deploy_gradio.git
cd ComfyUI_deploy_gradio
docker build -t comfy_ollama .
```

### 2️⃣ **Running Locally (Without Ollama, Gradio Only)**
```bash
cd ComfyUI_deploy_gradio/ComfyUI
conda create -n comfy_env
conda activate comfy_env
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install gradio requests
pip install -r /ComfyUI/requirements.txt
python demo.py
```

## 📂 **System Configuration**
### **1. Download Required Model Weights**
Before running the system, ensure that the necessary model weights are placed in the correct directories:

- **UNet Weights** (`ComfyUI/models/unet`)
  - [flux1-dev-fp8.safetensors](https://huggingface.co/lllyasviel/flux1_dev/blob/main/flux1-dev-fp8.safetensors)

- **VAE Weights** (`ComfyUI/models/vae`)
  - [ae.safetensors](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors)

- **Text Encoder Weights** (`ComfyUI/models/text_encoders`)
  - [clip_l.safetensors](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/clip_l.safetensors)
  - [t5xxl_fp16.safetensors](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors)

### **2. Docker**
- `Dockerfile`: Defines the entire Flux Dev environment.
- Build the image with `docker build -t comfy_ollama .`, then run it using:
  ```bash
  docker run -d --name comfy_ollama --gpus all -p 11434:11434 -p 7860:7860 comfy_ollama
  ```

### **3. Ngrok**
- Ngrok is used to expose the Gradio UI for remote access.
- Run Ngrok with: `ngrok http 7860`

### **4. Ollama (Korean Prompt Translation)**
- Uses the `EEVE-Korean-Instruct-10.8B` model to translate Korean prompts into English.

## 🖥 **Gradio UI Usage**
- Access the web UI at `http://localhost:7860` or the Ngrok-provided URL.
- Enter a prompt and click **"Generate Image"** to create an AI-generated image.

## 🛠 **Tech Stack**
- **Python 3.13.2**
- **Gradio 4.x**
- **Stable Diffusion 3 (Flux Dev-based)**
- **Docker**
- **Ngrok**
- **Ollama (Korean Translation)**

## 🚀 **Features & TODO**
✅ **Current Features:**
- Detects Korean prompts and translates them using Ollama
- Generates AI images using Stable Diffusion 3
- Allows users to request image generation via Gradio UI
- Supports external access using Ngrok

🛠 **Future Enhancements:**
- Improve image quality (Hypernetwork tuning)
- Add user-specific prompt saving & retrieval
- Optimize GPU performance for faster response times

---
📌 **Contributions & Support**
- Pull requests are welcome! 😊
- Report issues via [GitHub Issues](https://github.com/your-repo/flux-dev-ai/issues).

