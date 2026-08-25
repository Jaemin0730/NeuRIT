import torch

# ==========================================
# 1. 설정 (Soft Masking Weights)
# 요청하신 가중치 설정
# ==========================================

BOTTOM_W = 1.0

# ==========================================
# 2. 데이터 입력 (제공해주신 데이터)
# 형식: ["layer@neuron_idx", score/count]
# ==========================================


raw_bottom = [
    ["32@6503", 9], ["24@6640", 8], ["29@876", 7], ["34@2112", 7], ["34@9419", 6], ["31@3131", 6], ["34@9434", 5], ["23@9119", 5], ["33@8995", 3], ["30@7005", 3], ["24@3725", 3], ["35@4113", 3], ["31@2592", 3], ["31@5923", 3], ["34@11103", 3], ["31@3757", 3], ["32@1562", 3], ["31@213", 3], ["24@9192", 3], ["31@6411", 2], ["34@3577", 2], ["34@7418", 2], ["34@4274", 2], ["24@855", 2], ["25@7818", 2], ["32@4167", 2], ["4@10479", 2], ["32@1320", 2], ["32@4258", 2], ["34@5728", 2], ["30@8543", 2], ["24@577", 2], ["34@1183", 2], ["31@6406", 2], ["33@11650", 2], ["34@6605", 2], ["35@4908", 2], ["32@9832", 2], ["28@441", 2], ["32@7060", 2], ["32@6566", 2], ["34@2065", 2], ["24@4591", 2], ["28@3865", 2], ["25@8367", 2], ["31@2021", 1], ["34@61", 1], ["34@8993", 1], ["34@6962", 1], ["32@1130", 1], ["34@3018", 1], ["25@5588", 1], ["25@8905", 1], ["26@5522", 1], ["32@5229", 1], ["31@10438", 1], ["29@1106", 1], ["34@118", 1], ["34@2188", 1], ["35@5902", 1], ["31@9106", 1], ["29@1610", 1], ["23@2698", 1], ["34@3827", 1], ["34@136", 1], ["25@5898", 1], ["33@11281", 1], ["35@11141", 1], ["35@8835", 1], ["35@906", 1], ["35@163", 1], ["35@6430", 1], ["26@9384", 1], ["34@5352", 1], ["34@9129", 1], ["23@4293", 1], ["29@5209", 1], ["34@11051", 1], ["31@3540", 1], ["31@6512", 1], ["31@2285", 1], ["31@10014", 1], ["34@9668", 1], ["15@9466", 1], ["31@10662", 1], ["31@4637", 1], ["31@10360", 1], ["31@1242", 1], ["34@2497", 1], ["31@6552", 1], ["34@497", 1], ["32@10369", 1], ["21@533", 1], ["32@5163", 1], ["32@1944", 1], ["35@3926", 1], ["24@10710", 1], ["34@3217", 1], ["34@9341", 1], ["34@3204", 1]
]

# ==========================================
# 3. 모델 파라미터 (Qwen3-8B 기준)
# ==========================================
NUM_LAYERS = 36
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 12288  # Qwen3-8B의 FFN 차원

# ==========================================
# 4. 마스크 딕셔너리 초기화 (전부 0으로 시작)
# ==========================================
mask_dict = {}
print(f"Initializing masks for {NUM_LAYERS} layers (Target: Soft Masking)...")

for i in range(NUM_LAYERS):
    layer_name_in = f"{i}_in"
    layer_name_out = f"{i}_out"
    
    mask_dict[layer_name_in] = torch.zeros(HIDDEN_SIZE, INTERMEDIATE_SIZE)
    mask_dict[layer_name_out] = torch.zeros(INTERMEDIATE_SIZE, HIDDEN_SIZE)

print("Initialization complete.")

# ==========================================
# 5. Soft Masking 적용 함수
# ==========================================
def apply_mask_to_group(neuron_data_list, weight, group_name):
    """
    neuron_data_list: ["layer@neuron", count] 형태의 리스트
    weight: 적용할 가중치 (실수값)
    group_name: 로그 출력용 이름
    """
    applied_count = 0
    for item in neuron_data_list:
        # 데이터 파싱: "0@9187" -> layer=0, neuron=9187
        info_str = item[0] 
        layer_str, neuron_str = info_str.split('@')
        
        layer_idx = int(layer_str)
        neuron_idx = int(neuron_str)
        
        # 범위 체크
        if layer_idx >= NUM_LAYERS or neuron_idx >= INTERMEDIATE_SIZE:
            print(f"Warning: {group_name} index [L{layer_idx}, N{neuron_idx}] out of bounds. Skipping.")
            continue

        # up_proj ('_in'): 해당 뉴런의 열(column)에 weight 적용
        in_key = f"{layer_idx}_in"
        mask_dict[in_key][:, neuron_idx] = weight
        
        # down_proj ('_out'): 해당 뉴런의 행(row)에 weight 적용
        out_key = f"{layer_idx}_out"
        mask_dict[out_key][neuron_idx, :] = weight
        
        applied_count += 1
    
    print(f"Applied [{group_name}] group: {applied_count} neurons with weight {weight}")

# ==========================================
# 6. 그룹별 적용 실행
# ==========================================
print("\n--- Applying Soft Masks ---")
# apply_mask_to_group(raw_top, TOP_W, "TOP")
# apply_mask_to_group(raw_inter, INTER_W, "INTER")
apply_mask_to_group(raw_bottom, BOTTOM_W, "BOTTOM")

# ==========================================
# 7. 파일 저장
# ==========================================
save_path = 'sft_neuron_mask.pt'
torch.save(mask_dict, save_path)
print(f"\nSuccessfully saved SFT neuron mask to {save_path}")

# ==========================================
# 8. 검증 (값이 제대로 들어갔는지 확인)
# ==========================================
print("\n--- Verifying Mask Values ---")
loaded_dict = torch.load(save_path)

# 랜덤하게 몇 개 레이어의 max 값을 찍어보며 0, 0.5, 1.0이 존재하는지 확인
sample_layer = "30_in"  # Bottom, Inter가 많이 포함된 30번 레이어 확인
if sample_layer in loaded_dict:
    unique_values = torch.unique(loaded_dict[sample_layer])
    print(f"Unique values in {sample_layer}: {unique_values}")
    # 예상 결과: tensor([0.0000, 0.5000, 1.0000]) 등이 나와야 함 (0.5는 bottom, 1.0은 top/inter)

print("Verification complete! Ready for training.")