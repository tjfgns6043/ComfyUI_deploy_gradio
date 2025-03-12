import gradio as gr
import torch
import random
import numpy as np
import gc
import time
import csv
import os
import datetime
from PIL import Image

from comfy_extras.nodes_sd3 import EmptySD3LatentImage
from comfy_extras.nodes_custom_sampler import (
    BasicGuider, Noise_RandomNoise, KSamplerSelect, SamplerCustomAdvanced, BasicScheduler
)
from nodes import (
    UNETLoader, DualCLIPLoader, CLIPTextEncode, VAELoader, VAEDecode
)
from comfy_extras.nodes_flux import FluxGuidance
import requests
import threading

# =======================================
# 1) 글로벌 설정
# =======================================
PROCESSING_TIME_PER_PIXEL = 0.00005

active_requests = 0
lock = threading.Lock()

LOG_FILE = "generation_logs.csv"

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "is_korean", "prompt_length",
                         "width", "height", "steps", "seed", "guidance",
                         "estimated_time", "actual_time",
                         "original_prompt", "translated_prompt"])

def log_generation_data(is_korean, prompt, translated_prompt,
                        width, height, steps, seed, guidance, estimated_time, actual_time):
    timestamp = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    prompt_length = len(prompt)

    log_data = [
        timestamp,
        is_korean,
        prompt_length,
        width,
        height,
        steps,
        seed,
        guidance,
        estimated_time,
        actual_time,
        prompt,
        translated_prompt
    ]
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(log_data)

def translate_korean_to_english(prompt):
    instruction = (
        "Translate the following Korean text into natural, concise English. "
        "The translation must be optimized for text-to-image generation. "
        "Do NOT add explanations, comments, or extra words—only return the translated phrase.\n\n"
    ) 
    full_prompt = instruction + prompt
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "hf.co/teddylee777/EEVE-Korean-Instruct-10.8B-v1.0-gguf:Q8_0",
            "prompt": full_prompt,
            "stream": False
        }
    )
    if response.status_code == 200:
        torch.cuda.empty_cache()
        gc.collect()
        return response.json()["response"].strip()
    else:
        return "Translation failed."

def random_seed():
    return random.randint(1, 10**15 - 1)

def estimate_processing_time(width, height, steps):
    width = width if width else 512
    height = height if height else 512
    steps = steps if steps else 25

    total_pixels = width * height
    steps_boost = (steps / 25)
    estimated_time = total_pixels * PROCESSING_TIME_PER_PIXEL * steps_boost
    return round(estimated_time, 2)

# =======================================
# 2) 전역 모델 (UNet, CLIP, VAE) 로드
# =======================================
print("Loading models once at startup...")

# 2A. UNet 로드
unet_loader = UNETLoader()
unet_name = "flux1-dev-fp8.safetensors"
unet_model = unet_loader.load_unet(unet_name, "default")[0]

# 2B. CLIP 로드
clip_loader = DualCLIPLoader()
clip_model = clip_loader.load_clip("clip_l.safetensors", "t5xxl_fp16.safetensors", "flux")[0]
text_encoder = CLIPTextEncode()

# 2C. VAE 로드
vae_loader = VAELoader()
vae_model = vae_loader.load_vae("ae.safetensors")[0]
vae_decoder = VAEDecode()

print("✅ All models loaded successfully!")

# =======================================
# 3) 이미지 생성 함수
# =======================================
def generate_image(prompt, width, height, seed, guidance, steps):
    global active_requests, unet_model, clip_model, vae_model

    with lock:
        active_requests += 1

    torch.cuda.empty_cache()
    start_time = time.time()

    # (1) 한국어인지 감지
    is_korean = any("가" <= char <= "힣" for char in prompt)

    # (2) 번역
    translated_prompt = prompt
    if is_korean:
        translated_prompt = translate_korean_to_english(prompt)
        print(translated_prompt)

    # (3) 예상 처리 시간
    estimated_time = estimate_processing_time(width, height, steps)

    # ==== Diffusion 파이프라인 ====    
    # Latent Image 생성
    latent_generator = EmptySD3LatentImage()
    latent_image = latent_generator.generate(width, height, 1)[0]

    # Noise 추가
    noise_generator = Noise_RandomNoise(seed)
    noisy_latent = noise_generator.generate_noise(latent_image)

    # 샘플러 선택
    sampler_selector = KSamplerSelect()
    sampler_name = "euler"
    selected_sampler = sampler_selector.get_sampler(sampler_name)[0]

    # 스케줄러 설정
    scheduler = BasicScheduler()
    sigmas = scheduler.get_sigmas(unet_model, "simple", steps, 1.0)[0]

    # 텍스트 인코딩
    encoded_text = text_encoder.encode(clip_model, translated_prompt)[0]

    # Flux 가이던스
    flux_guidance = FluxGuidance()
    conditioned_text = flux_guidance.append(encoded_text, guidance)[0]

    # Guider 생성
    basic_guider = BasicGuider()
    guider = basic_guider.get_guider(unet_model, conditioned_text)[0]

    # 샘플링 실행
    sampler_advanced = SamplerCustomAdvanced()
    sampled_latent, _ = sampler_advanced.sample(
        noise_generator, guider, selected_sampler, sigmas, latent_image
    )

    # VAE 디코딩
    decoded_image = vae_decoder.decode(vae_model, sampled_latent)[0]
    decoded_image_np = (decoded_image[0].cpu().detach().numpy() * 255).astype(np.uint8)
    image_pil = Image.fromarray(decoded_image_np)

    with lock:
        active_requests -= 1

    total_time = round(time.time() - start_time, 2)

    # (4) 로깅 (seed 포함)
    log_generation_data(
        is_korean, prompt, translated_prompt,
        width, height, steps, seed, guidance,
        estimated_time, total_time
    )

    return image_pil

# =======================================
# 4) 비율 버튼 콜백
# =======================================
def set_aspect_ratio(ratio):
    if ratio == "256:256":
        w, h = 256, 256
    elif ratio == "512:512":
        w, h = 512, 512
    elif ratio == "832:1216":
        w, h = 832, 1216
    elif ratio == "768:1344":
        w, h = 768, 1344
    else:
        w, h = 256, 256

    # steps=25로 가정
    steps_default = 25
    est_time = estimate_processing_time(w, h, steps_default)
    return (w, h, est_time)

# =======================================
# 5) Gradio 인터페이스
# =======================================
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            # Flux.1 Dev
            Flux.1 Dev는 **AI 기반 이미지 생성 모델**로, 입력한 텍스트(프롬프트)에 따라 창의적인 이미지를 생성하는 최신 AI입니다.  
            Stable Diffusion의 발전된 버전이며, **더 빠르고, 더 선명한 이미지**를 만들어냅니다.  

            ### 🔗 Flux에 대해 더 알고 싶다면?
            Flux.1 Dev의 원리, 연구 내용, 최신 업데이트 정보는 아래 공식 블로그에서 확인할 수 있습니다.  
            [Flux Blog](https://blackforestlabs.ai/announcing-black-forest-labs/)
            """)

        with gr.Column():
            gr.Markdown("""
            ### **🔹 Guidance Scale이란?**
            **Guidance Scale**은 AI가 입력한 텍스트(프롬프트)를 **얼마나 중요하게 여길지** 결정하는 매개변수입니다.  

            ✔ **값이 낮으면**  
            → AI가 더 자유롭게 창작하며, 창의적인 결과물이 나올 수 있습니다.  
            ✔ **값이 높으면**  
            → 입력한 문장(프롬프트)과 최대한 일치하도록 이미지를 생성합니다.  

            💡 **Tip:**  
            - 🎨 창의적인 그림이 필요하다면 낮은 값(예: 2~4)  
            - 📸 입력 문장과 최대한 일치하는 이미지가 필요하면 높은 값(예: 8~10)  
            """)

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", value="A futuristic city with neon lights")
            width = gr.Slider(0, 1600, value=256, step=32, label="Width")
            height = gr.Slider(0, 1600, value=256, step=32, label="Height")

            with gr.Row():
                aspect_ratio_s1_1 = gr.Button("256:256")
                aspect_ratio_1_1 = gr.Button("512:512")
                aspect_ratio_16_9 = gr.Button("832:1216")
                aspect_ratio_4_3 = gr.Button("768:1344")

            seed = gr.Number(value=random_seed(), label="Seed")
            seed_button = gr.Button("Randomize Seed")
            guidance = gr.Slider(1.0, 10.0, value=3.5, step=0.1, label="Guidance Scale")
            steps = gr.Slider(1, 100, value=25, step=1, label="Sampling Steps")

            estimated_time = gr.Textbox(label="Estimated Wait Time (seconds)", interactive=False)

            generate_button = gr.Button("Generate Image")

        with gr.Column():
            output_image = gr.Image(label="Generated Image")

    # 슬라이더 변경 => 예상 대기 시간 업데이트
    def update_estimated_time(w, h, s):
        return estimate_processing_time(w, h, s)

    width.change(update_estimated_time, inputs=[width, height, steps], outputs=[estimated_time], queue=False)
    height.change(update_estimated_time, inputs=[width, height, steps], outputs=[estimated_time], queue=False)
    steps.change(update_estimated_time, inputs=[width, height, steps], outputs=[estimated_time], queue=False)

    # 비율 버튼 => Width, Height, EstimatedTime
    aspect_ratio_s1_1.click(fn=lambda: set_aspect_ratio("256:256"), outputs=[width, height, estimated_time], queue=False)
    aspect_ratio_1_1.click(fn=lambda: set_aspect_ratio("512:512"),   outputs=[width, height, estimated_time], queue=False)
    aspect_ratio_16_9.click(fn=lambda: set_aspect_ratio("832:1216"), outputs=[width, height, estimated_time], queue=False)
    aspect_ratio_4_3.click(fn=lambda: set_aspect_ratio("768:1344"),   outputs=[width, height, estimated_time], queue=False)

    seed_button.click(fn=random_seed, outputs=[seed])

    def wrapper_generate_image(p, w, h, sd, g, st):
        return generate_image(p, w, h, sd, g, st)

    generate_button.click(
        fn=wrapper_generate_image,
        inputs=[prompt, width, height, seed, guidance, steps],
        outputs=[output_image],
        queue=True,
        concurrency_limit=1
    )

if __name__ == "__main__":
    demo.launch()
