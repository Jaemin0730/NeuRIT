model_path="meta-llama/Meta-Llama-3-8B-Instruct"
model_name="llama3-8B-instruct"

# 2단계 산출물 (샘플별 (layer, neuron)[, score] 목록)
top_cn_dir="results/cn/4/topbottom-llama3-8B-top_cn_bag.json"
inter_cn_dir="results/cn/4/topbottom-llama3-8B-intersection_cn_bag.json"
bottom_cn_dir="results/cn/4/topbottom-llama3-8B-bottom_cn_bag.json"

cn_dir="results/cn/4"

# 탐색 구간
# KS : neuron 개수
# SS : strength
TOP_KS=(100)
INTER_KS=(30)
BOTTOM_KS=(100)

TOP_SS=(10)                 # 정수 OK
INTER_SS=(2)   # 1 2 3 ... 10
BOTTOM_SS=(10)  # 0.0 0.1 ... 0.9


# TOP_KS=(12)              # 정수 한 개              
# INTER_SS=(0.7)           # 실수 한 개
# TOP_KS=(4 8 12)          # 정수 여러 개
# INTER_SS=(0.3 0.5 0.7)   # 실수 여러 개

# TOP_KS=({1..20})
# INTER_KS=({1..20})
# BOTTOM_KS=({1..20})
# TOP_SS=({2..16})                 # 정수 OK
# INTER_SS=($(seq 0.1 0.1 0.9))    # 실수는 seq
# BOTTOM_SS=($(seq 0.0 0.1 0.5))


data_paths=("data/top389.jsonl")
out_root="eval_results/44/outputs"
metric_root="eval_results/44/metrics"

mkdir -p "$out_root" "$metric_root"
echo "Model: $model_path ($model_name)"

for data_path in "${data_paths[@]}"; do
  ds_tag="$(basename "${data_path%.jsonl}")"   # topk or bottomk

  for tk in "${TOP_KS[@]}"; do
    for ik in "${INTER_KS[@]}"; do
      for bk in "${BOTTOM_KS[@]}"; do
        for ts in "${TOP_SS[@]}"; do
          for is in "${INTER_SS[@]}"; do
            for bs in "${BOTTOM_SS[@]}"; do
              tag="${ds_tag}__tk${tk}_ik${ik}_bk${bk}__ts${ts}_is${is}_bs${bs}"
              echo ">> [RUN] $tag"

              python src/3_enhance_and_evaluate_topbottom.py \
                --model_path "$model_path" \
                --model_name "$model_name" \
                --data_path "$data_path" \
                --dataset_name topbottom \
                --max_seq_length 512 \
                --cn_dir "$cn_dir" \
                --top_cn_dir "$top_cn_dir" \
                --inter_cn_dir "$inter_cn_dir" \
                --bottom_cn_dir "$bottom_cn_dir" \
                --enhance_cn_num_top "$tk" \
                --enhance_cn_num_inter "$ik" \
                --enhance_cn_num_bottom "$bk" \
                --enhance_strength_top "$ts" \
                --enhance_strength_inter "$is" \
                --enhance_strength_bottom "$bs" \
                --output_dir "${out_root}" \
                --metric_dir "${metric_root}"
            done
          done
        done
      done
    done
  done
done