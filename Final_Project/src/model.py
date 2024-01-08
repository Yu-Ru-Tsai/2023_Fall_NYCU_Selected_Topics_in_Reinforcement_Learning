import torch

# 載入模型權重
load_path = './log/austria_competition/test15/model_3323978_0.424.pth'
checkpoint = torch.load(load_path)

# 獲取模型結構的名稱
model_structure = checkpoint['model_structure']
print("Model Structure:")
print(model_structure)
