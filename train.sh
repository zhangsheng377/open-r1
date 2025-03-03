export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/ddp.yaml --num_processes=1 src/open_r1/grpo.py --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml 2>&1 | tee log.log

# ollama create zhangsheng377/qwen2.5-7b-instruct-r1-lora -f /mnt/nfs/zsd_server/codes/open-r1/ModelFile.txt
