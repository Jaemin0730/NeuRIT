python src/2_get_cns_topbottom.py \
  --result_dir results/attribution \
  --cn_dir results/cn/2 \
  --top_file 2/llama3-8b/top-llama3-8B-context.rlt.jsonl \
  --bottom_file 2/llama3-8b/bottom-llama3-8B-context.rlt.jsonl \
  --dataset_name topbottom \
  --model_name llama3-8B-instruct \
  --top_k 20