import os
import json
import argparse
from tqdm import tqdm

def stat(cn_bag_list, pos_type, tag):
    if not cn_bag_list:
        print(f"{tag}'s {pos_type} has 0 items.")
        return
    ave_len = sum(len(bag) for bag in cn_bag_list) / len(cn_bag_list)
    print(f"{tag}'s {pos_type} has on average {ave_len:.2f} imp pos.")

def analysis_context_file(filepath, metric='all_attr_gold', top_k=20):
    """
    JSONL 파일에서 각 샘플의 metric 리스트(= triplet 리스트)를 내려받아
    score 내림차순으로 정렬 후 상위 top_k만 남겨 반환.
    triplet은 [layer, neuron, score] 형태를 가정.
    """
    print(f"===========> parsing important position in {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    rlts_bag = []
    with open(filepath, 'r') as fr:
        for idx, line in enumerate(tqdm(fr.readlines())):
            try:
                example = json.loads(line)
                rlts_bag.append(example)
            except Exception as e:
                print(f"Exception {e} happened in line {idx}")

    print(f"Total examples: {len(rlts_bag)}")

    cn_bag_list = []
    for rlt in rlts_bag:
        metric_triplets = rlt.get(metric, [])
        try:
            metric_triplets.sort(key=lambda x: x[2], reverse=True)
        except Exception as e:
            print(f"Skip an example due to format error: {e}")
            continue
        cn_bag_list.append(metric_triplets[:top_k])  # 정확히 top_k개 유지
    return cn_bag_list

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fw:
        json.dump(obj, fw, indent=2, ensure_ascii=False)
    print(f"Saved: {path}")

def build_prefix(args):
    """
    파일명 앞에 붙일 prefix를 만듭니다.
    예) topbottom-llama3-8B
    """
    parts = []
    if args.dataset_name:
        parts.append(args.dataset_name)
    if args.model_name:
        parts.append(args.model_name)
    return "-".join(parts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Directory containing the input JSONL files.")
    parser.add_argument("--cn_dir", type=str, required=True,
                        help="Directory to write the output JSON files.")
    # 두 입력 파일을 직접 지정
    parser.add_argument("--top_file", type=str, required=True,
                        help="Filename of the TOP JSONL (relative to result_dir).")
    parser.add_argument("--bottom_file", type=str, required=True,
                        help="Filename of the BOTTOM JSONL (relative to result_dir).")

    parser.add_argument("--top_k", type=int, default=20,
                        help="Per-sample number of neurons to keep before splitting.")
    parser.add_argument("--metric", type=str, default="all_attr_gold",
                        help="Key name in each JSON line that holds triplets.")
    
    parser.add_argument("--dataset_name", type=str, required=False, default=None)
    parser.add_argument("--model_name", type=str, required=False, default=None)

    args = parser.parse_args()

    os.makedirs(args.cn_dir, exist_ok=True)

    top_path = os.path.join(args.result_dir, args.top_file)
    bottom_path = os.path.join(args.result_dir, args.bottom_file)

    # setting : 상위 top_k 추출
    top_cn_bag_list = analysis_context_file(top_path, metric=args.metric, top_k=args.top_k)
    bottom_cn_bag_list = analysis_context_file(bottom_path, metric=args.metric, top_k=args.top_k)

    # 0) 전역 교집합 키 계산
    all_top    = [t for bag in top_cn_bag_list for t in bag if len(t) >= 2]
    all_bottom = [t for bag in bottom_cn_bag_list for t in bag if len(t) >= 2]

    top_pairs_all    = {(t[0], t[1]) for t in all_top}
    bottom_pairs_all = {(t[0], t[1]) for t in all_bottom}
    inter_pairs      = top_pairs_all & bottom_pairs_all   # 전역 inter 키 집합

    # 1) inter triplet은 "top 쪽 점수 보존" 정책으로 구성
    intersection_list = [t for t in all_top if (t[0], t[1]) in inter_pairs]

    # 2) 각 샘플별로 inter 제거한 리스트 재구성 (샘플 구조 보존)
    top_only_list = []
    bottom_only_list = []
    for top_bag, bottom_bag in zip(top_cn_bag_list, bottom_cn_bag_list):
        top_only    = [t for t in top_bag    if len(t) >= 2 and (t[0], t[1]) not in inter_pairs]
        bottom_only = [t for t in bottom_bag if len(t) >= 2 and (t[0], t[1]) not in inter_pairs]
        top_only_list.append(top_only)
        bottom_only_list.append(bottom_only)

    # 3) 통계 출력
    stat(top_cn_bag_list, 'cn_bag(original)', 'top')
    stat(bottom_cn_bag_list, 'cn_bag(original)', 'bottom')
    stat(top_only_list, 'cn_bag', 'top_only')
    stat(bottom_only_list, 'cn_bag', 'bottom_only')
    stat(intersection_list, 'cn_bag', 'intersection')

    # 3-1) 총 고유 뉴런(=고유 (layer, neuron)) 개수 요약
    top_only_pairs    = {(t[0], t[1]) for bag in top_only_list    for t in bag}
    bottom_only_pairs = {(t[0], t[1]) for bag in bottom_only_list for t in bag}
    print("=" * 60)
    print("[Unique candidate neurons | unique (layer, neuron)]")
    print(f"  relevant   (top    candidates): {len(top_pairs_all)}")
    print(f"  irrelevant (bottom candidates): {len(bottom_pairs_all)}")
    print(f"  overlap    (intersection)     : {len(inter_pairs)}")
    print(f"  union      (top | bottom)     : {len(top_pairs_all | bottom_pairs_all)}")
    print("-" * 60)
    print(f"  after split -> top_only     : {len(top_only_pairs)}")
    print(f"  after split -> bottom_only  : {len(bottom_only_pairs)}")
    print(f"  after split -> intersection : {len(inter_pairs)}")
    print("=" * 60)

    # 3-2) 중복 포함(고유 처리 X) 뉴런 개수 요약 = 등장한 (layer, neuron) 총 개수
    top_only_total    = sum(len(bag) for bag in top_only_list)
    bottom_only_total = sum(len(bag) for bag in bottom_only_list)
    print("[Candidate neurons WITH duplicates | total occurrences]")
    print(f"  relevant   (top    candidates): {len(all_top)}")
    print(f"  irrelevant (bottom candidates): {len(all_bottom)}")
    print(f"  overlap    (intersection)     : {len(intersection_list)}")
    print(f"  union      (top + bottom)     : {len(all_top) + len(all_bottom)}")
    print("-" * 60)
    print(f"  after split -> top_only     : {top_only_total}")
    print(f"  after split -> bottom_only  : {bottom_only_total}")
    print(f"  after split -> intersection : {len(intersection_list)}")
    print("=" * 60)

    # 4) 저장 (서로 배타적 분할 결과)
    prefix = build_prefix(args)  # ← prefix 생성

    save_json(top_only_list,     os.path.join(args.cn_dir, f"{prefix}-top_cn_bag-context.json"))
    save_json(intersection_list, os.path.join(args.cn_dir, f"{prefix}-intersection_cn_bag-context.json"))
    save_json(bottom_only_list,  os.path.join(args.cn_dir, f"{prefix}-bottom_cn_bag-context.json"))