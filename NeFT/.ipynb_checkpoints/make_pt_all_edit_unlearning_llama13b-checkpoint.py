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
    ["38@10389", 43], ["12@12004", 28], ["10@12415", 24], ["24@10127", 18], ["25@1065", 18], ["9@11943", 17], ["11@10946", 13], ["13@3415", 12], ["10@236", 11], ["38@10401", 10], ["10@5538", 9], ["15@10270", 9], ["16@13049", 9], ["30@3460", 8], ["33@7282", 8], ["20@3039", 8], ["8@13288", 8], ["37@4711", 8], ["16@6277", 7], ["15@7485", 7], ["37@7051", 7], ["35@303", 7], ["7@1401", 7], ["35@1817", 6], ["15@10413", 6], ["9@2167", 6], ["10@9013", 6], ["10@3632", 5], ["10@5405", 5], ["19@10236", 5], ["20@5072", 5], ["13@5091", 5], ["20@4285", 5], ["33@5135", 4], ["10@7812", 4], ["11@13791", 4], ["29@11064", 4], ["10@5303", 4], ["9@11962", 4], ["38@1", 4], ["39@10017", 4], ["36@10292", 4], ["10@8228", 4], ["17@416", 4], ["37@12173", 4], ["17@4362", 3], ["7@992", 3], ["9@1922", 3], ["13@8937", 3], ["13@5775", 3], ["11@3281", 3], ["35@1446", 3], ["38@11620", 3], ["12@12758", 3], ["14@144", 3], ["28@6676", 3], ["8@2757", 3], ["22@9916", 3], ["36@10753", 3], ["13@2506", 3], ["8@9812", 3], ["8@2140", 3], ["15@13147", 3], ["10@5821", 3], ["36@6227", 3], ["10@9589", 3], ["12@6903", 2], ["9@2407", 2], ["12@7140", 2], ["20@4894", 2], ["39@9308", 2], ["10@10748", 2], ["21@10655", 2], ["9@4461", 2], ["28@9906", 2], ["36@11120", 2], ["16@7075", 2], ["36@8243", 2], ["12@11595", 2], ["19@4334", 2], ["18@7690", 2], ["11@5271", 2], ["38@13241", 2], ["14@9916", 2], ["22@9833", 2], ["13@9801", 2], ["10@6736", 2], ["9@2602", 2], ["20@3423", 2], ["20@3559", 2], ["12@8635", 2], ["20@8911", 2], ["9@11503", 2], ["20@1792", 2], ["15@8257", 2], ["11@1186", 2], ["8@4560", 2], ["37@9539", 2], ["8@10753", 2], ["10@3571", 2]
]

# ==========================================
# 3. 모델 파라미터 (Llama 2 13B 기준)
# ==========================================
NUM_LAYERS = 40
HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 13824 # 13B 모델의 FFN 차원

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