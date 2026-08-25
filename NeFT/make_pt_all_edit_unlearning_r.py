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
    ["1@12565", 78],
    ["19@890", 367],
    ["5@12027", 181],
    ["22@3989", 117],
    ["15@2239", 86],
    ["28@13370", 351],
    ["7@14268", 386],
    ["20@2356", 206],
    ["30@1680", 94],
    ["17@10130", 225],
    ["8@2699", 234],
    ["26@11532", 223],
    ["1@5966", 212],
    ["9@1486", 111],
    ["10@11527", 301],
    ["4@8711", 153],
    ["21@9787", 400],
    ["18@11825", 299],
    ["16@11363", 91],
    ["21@1833", 400],
    ["11@13371", 255],
    ["10@9929", 353],
    ["0@9327", 120],
    ["7@14047", 395],
    ["31@12184", 7],
    ["14@9472", 348],
    ["11@697", 225],
    ["15@9077", 220],
    ["6@9042", 308],
    ["13@9901", 14],
    ["22@2913", 342],
    ["29@4788", 195],
    ["26@9629", 295],
    ["16@7093", 218],
    ["21@8008", 84],
    ["19@4301", 23],
    ["26@3667", 264],
    ["10@6819", 359],
    ["20@13016", 388],
    ["29@6454", 44],
    ["7@11098", 193],
    ["5@4348", 131],
    ["5@13858", 176],
    ["21@13711", 152],
    ["7@6677", 324],
    ["10@8265", 301],
    ["29@10136", 77],
    ["20@14117", 75],
    ["24@7799", 211],
    ["18@8613", 140],
    ["12@14072", 370],
    ["12@5642", 74],
    ["24@1262", 358],
    ["0@12428", 238],
    ["14@13756", 1],
    ["28@4313", 373],
    ["6@841", 242],
    ["12@11124", 368],
    ["30@4444", 135],
    ["5@8844", 152],
    ["8@11654", 285],
    ["2@2049", 270],
    ["0@1694", 233],
    ["20@1035", 95],
    ["6@11051", 169],
    ["23@12787", 206],
    ["28@7288", 233],
    ["22@14161", 240],
    ["14@13676", 151],
    ["20@12604", 89],
    ["22@9649", 215],
    ["31@784", 347],
    ["28@12433", 159],
    ["19@7638", 342],
    ["14@8346", 387],
    ["29@7356", 259],
    ["16@8551", 18],
    ["12@9934", 315],
    ["13@5361", 113],
    ["14@7448", 22],
    ["0@1247", 205],
    ["21@12278", 133],
    ["27@8058", 249],
    ["31@12418", 231],
    ["6@10312", 139],
    ["22@6356", 302],
    ["4@2268", 311],
    ["24@11607", 211],
    ["29@12739", 118],
    ["14@11011", 66],
    ["16@5035", 84],
    ["19@6243", 353],
    ["15@10376", 333],
    ["27@8388", 375],
    ["18@11812", 68],
    ["0@10013", 136],
    ["24@8071", 78],
    ["13@10675", 145],
    ["23@6889", 330],
    ["19@1845", 103],
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