import torch
from torchvision import transforms
from model import Generator  # Assuming you have the Generator model defined in model.py
import numpy as np
from PIL import Image

# 加载预训练模型
g = Generator(1024, 512, 8)  # 初始化生成器
checkpoint = torch.load('path/to/checkpoint.pt')
g.load_state_dict(checkpoint['g_ema'], strict=False)
g.eval()

# 生成批量人脸
def generate_faces(batch_size, output_dir):
    for i in range(batch_size):
        z = torch.randn(1, 512)  # 随机噪声向量
        img = g(z)
        img = (img.clamp(-1, 1) + 1) / 2.0  # 将像素值归一化到[0, 1]
        img = img.mul(255).byte()
        img = img[0].permute(1, 2, 0).numpy()  # 调整维度
        img = Image.fromarray(img)
        img.save(f"{output_dir}/face_{i}.png")

# 执行生成
generate_faces(10, 'output/faces')  # 生成10张人脸图片并保存到cocoduang/coco1目录
