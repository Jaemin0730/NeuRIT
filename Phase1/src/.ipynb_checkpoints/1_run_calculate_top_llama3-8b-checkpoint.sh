
echo "[INFO] Running TOPK"
python src/1_calculate_attribution_topbottom.py \
    --model_path meta-llama/Llama-2-13b-chat-hf \
    --data_path data/top389.jsonl \
    --model_name llama3-8B \
    --dataset_name top \
    --output_dir results/attribution/5/llama3-8b \
    --gpu_id 0 \
    --max_seq_length 512 \
    --batch_size 20
echo "[INFO] Running FINISHED TOPK"
echo "[DONE] Outputs saved. FISNISHED!!"