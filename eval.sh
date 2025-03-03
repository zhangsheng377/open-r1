MODEL=/mnt/nfs/zsd_server/codes/open-r1/output/Qwen2.5-7B-Open-R1-GRPO_merged
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=1024,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:1024,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/Qwen2.5-7B-Open-R1-GRPO_merged

# AIME 2024
#TASK=aime24
#lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
#    --custom-tasks src/open_r1/evaluate.py \
#    --use-chat-template \
#    --output-dir $OUTPUT_DIR

# MATH-500
TASK=math_500
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    2>&1 | tee eval_log.log

# GPQA Diamond
#TASK=gpqa:diamond
#lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
#    --custom-tasks src/open_r1/evaluate.py \
#    --use-chat-template \
#    --output-dir $OUTPUT_DIR

# LiveCodeBench
#lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
#    --use-chat-template \
#    --output-dir $OUTPUT_DIR