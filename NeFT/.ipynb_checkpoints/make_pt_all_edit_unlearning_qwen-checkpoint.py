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
    ["32@5375", 32], ["31@716", 27], ["31@10361", 17], ["32@11750", 15], ["32@6343", 15], ["32@118", 14], ["35@12572", 14], ["31@1302", 13], ["30@9058", 13], ["32@11356", 11], ["46@12536", 11], ["32@4497", 10], ["37@9204", 10], ["32@8514", 10], ["38@184", 10], ["32@8747", 8], ["36@5360", 8], ["37@6413", 8], ["36@1403", 8], ["47@6023", 8], ["32@6099", 7], ["32@2699", 7], ["28@10781", 7], ["28@700", 6], ["35@221", 6], ["38@7831", 6], ["33@2023", 6], ["32@10214", 5], ["32@6766", 5], ["32@13556", 5], ["37@6079", 5], ["36@4146", 5], ["32@9135", 5], ["32@12038", 5], ["33@8107", 4], ["29@529", 4], ["46@10127", 4], ["42@4177", 4], ["31@1536", 4], ["34@9882", 4], ["31@10856", 4], ["31@12667", 4], ["32@6720", 4], ["34@8421", 4], ["36@2454", 4], ["37@11063", 4], ["41@12226", 4], ["32@12456", 4], ["36@5882", 4], ["32@5197", 4], ["32@1293", 4], ["35@10607", 4], ["36@10414", 4], ["32@5517", 4], ["37@6491", 3], ["32@3569", 3], ["32@7100", 3], ["32@12090", 3], ["31@11159", 3], ["28@12685", 3], ["34@2396", 3], ["46@13772", 3], ["31@6241", 3], ["32@8807", 3], ["32@5302", 3], ["28@13410", 3], ["47@4740", 3], ["34@9548", 3], ["28@10841", 3], ["36@1327", 3], ["38@619", 3], ["32@206", 3], ["32@7425", 3], ["33@12986", 3], ["43@3459", 3], ["47@6101", 3], ["47@12570", 3], ["32@12840", 3], ["36@9353", 3], ["33@5231", 3], ["37@9465", 3], ["47@854", 3], ["36@2315", 3], ["38@9170", 3], ["40@2658", 3], ["32@2102", 3], ["33@12301", 2], ["34@948", 2], ["30@11672", 2], ["30@7235", 2], ["32@2086", 2], ["31@9137", 2], ["36@4159", 2], ["31@11021", 2], ["36@386", 2], ["46@9374", 2], ["28@2979", 2], ["42@4369", 2], ["32@3832", 2], ["29@10503", 2]
]

# ==========================================
# 3. 모델 파라미터 (Qwen2.5 14B Chat)
# ==========================================
NUM_LAYERS = 48
HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 13824

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