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
    ["31@2398", 21], ["31@5966", 14], ["31@8959", 14], ["15@2150", 13], ["31@6614", 10], ["30@10725", 6], ["23@12414", 6], ["19@377", 5], ["16@3176", 5], ["22@6505", 5], ["7@13106", 4], ["13@4208", 4], ["16@11581", 4], ["30@6083", 4], ["14@6308", 4], ["12@13848", 3], ["31@127", 3], ["13@2402", 3], ["31@4015", 3], ["6@6991", 3], ["20@11668", 3], ["30@3382", 3], ["24@1255", 3], ["12@7039", 2], ["13@6104", 2], ["8@11821", 2], ["14@9832", 2], ["20@8651", 2], ["31@12525", 2], ["18@1714", 2], ["31@1739", 2], ["14@9515", 2], ["11@13737", 2], ["3@9905", 2], ["16@5720", 2], ["15@9296", 2], ["14@416", 2], ["22@13863", 2], ["30@9708", 2], ["30@9387", 2], ["18@12158", 2], ["24@2711", 2], ["6@2322", 1], ["30@1795", 1], ["17@8935", 1], ["30@4382", 1], ["23@3976", 1], ["29@731", 1], ["15@5572", 1], ["31@3694", 1], ["8@7378", 1], ["31@7104", 1], ["4@8239", 1], ["6@9476", 1], ["31@13801", 1], ["27@2343", 1], ["31@10699", 1], ["3@7026", 1], ["19@12164", 1], ["15@6658", 1], ["15@3548", 1], ["31@12766", 1], ["10@10116", 1], ["31@10100", 1], ["30@9928", 1], ["12@7352", 1], ["10@5220", 1], ["19@995", 1], ["13@3573", 1], ["5@1529", 1], ["31@4501", 1], ["30@8539", 1], ["16@458", 1], ["4@5812", 1], ["21@10723", 1], ["13@2940", 1], ["17@354", 1], ["28@10749", 1], ["31@311", 1], ["30@6816", 1], ["11@2766", 1], ["9@486", 1], ["31@12968", 1], ["30@4350", 1], ["6@12968", 1], ["24@1323", 1], ["31@3917", 1], ["22@1296", 1], ["31@1399", 1], ["28@1648", 1], ["12@7354", 1], ["9@843", 1], ["11@13112", 1], ["12@6697", 1], ["28@7549", 1], ["29@14171", 1], ["15@3359", 1], ["12@5275", 1], ["8@7471", 1], ["31@1678", 1]
]

# ==========================================
# 3. 모델 파라미터 (Llama 3 8B 기준)
# ==========================================
NUM_LAYERS = 32
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 14336 # 8B 모델의 FFN 차원

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