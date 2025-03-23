
MODEL_PATH="/mnt/longcontext/models/siyuan/llama3/Qwen2.5-VL-72B-Instruct/"
vllm serve ${MODEL_PATH} --api-key token-abc123 --tensor-parallel-size 4 --gpu-memory-utilization 0.9  --trust-remote-code --max_model_len 32768 --port 8000 | tee -a vllm.log