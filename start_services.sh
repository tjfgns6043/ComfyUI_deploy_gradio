#!/bin/bash

# ✅ Ollama 실행 (백그라운드 실행)
export OLLAMA_NO_CUDA=1
ollama serve &

# ✅ Ollama가 실행될 때까지 대기 후 모델 다운로드
sleep 5
ollama pull hf.co/teddylee777/EEVE-Korean-Instruct-10.8B-v1.0-gguf:Q8_0

# ✅ Gradio 실행 (백그라운드)
cd /app/ComfyUI/
python demo.py &

# ✅ Ngrok 실행 (백그라운드) + 로그 저장
ngrok http 7860 > /app/ngrok.log 2>&1 &

# ✅ Ngrok이 정상적으로 실행될 때까지 URL 가져오기
NGROK_URL=""
for i in {1..10}; do
    NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | jq -r ".tunnels[0].public_url")
    if [[ "$NGROK_URL" != "null" && "$NGROK_URL" != "" ]]; then
        break
    fi
    sleep 1
done

# ✅ 결과 출력 및 로그 저장
if [[ "$NGROK_URL" != "null" && "$NGROK_URL" != "" ]]; then
    echo "✅ ngrok is running!"
    echo "🔗 ngrok public URL: $NGROK_URL"
    echo "$NGROK_URL" > /app/ngrok_url.txt
else
    echo "❌ Failed to get ngrok URL. Check logs."
    cat /app/ngrok.log
fi

# ✅ 도커가 종료되지 않도록 포그라운드 유지
tail -f /dev/null
