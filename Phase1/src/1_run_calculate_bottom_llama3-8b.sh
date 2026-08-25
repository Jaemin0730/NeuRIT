
echo "[INFO] Running BOTTOMK"
python src/1_calculate_attribution_topbottom.py \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --data_path data/bottom389.jsonl \
    --model_name llama3-8B \
    --dataset_name bottom \
    --output_dir results/attribution/llama3-8b \
    --gpu_id 0 \
    --max_seq_length 512 \
    --batch_size 20
echo "[INFO] Running FINISHED BOTTOMK"
echo "[DONE] Outputs saved. FISNISHED!!"