"""
Flux + Visual Context Adapter 推理脚本

功能:
1. 加载模型与Adapter
2. 读取输入的骨架图
3. 执行生成
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flux.model import Flux, FluxParams
from flux.modules.adapter import VisualContextAdapter
from flux.modules.autoencoder import AutoEncoder
from flux.modules.conditioner import HFEmbedder
from flux.sampling import get_noise, prepare
# 假设有一个 denoise 函数在 sampling 中，或者我们需要自己实现 loop
# from flux.sampling import denoise 

def load_image(path, target_size=None):
    img = Image.open(path).convert("RGB")
    if target_size:
        img = img.resize(target_size, Image.Resampling.LANCZOS)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--skeleton_path", type=str, required=True, help="Path to skeleton image")
    parser.add_argument("--output_path", type=str, default="output.png")
    parser.add_argument("--model_path", type=str, default="models/flux-dev")
    parser.add_argument("--adapter_path", type=str, default="models/adapter.pt")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    dtype = torch.bfloat16
    
    # 1. 加载模型 components
    print("Loading models...")
    # 这里需要根据具体Flux实现加载各个组件 (T5, CLIP, AE, Flux)
    # mock loading for structure
    t5 = HFEmbedder("google/t5-v1_1-xxl").to(device)
    clip = HFEmbedder("openai/clip-vit-large-patch14").to(device)
    ae = AutoEncoder().to(device) # load weights logic needed
    
    # Load Flux
    # flux = Flux.load(args.model_path).to(device)
    flux_params = FluxParams(
        in_channels=64, out_channels=64, vec_in_dim=768, context_in_dim=4096,
        hidden_size=3072, mlp_ratio=4.0, num_heads=24, depth=19, depth_single_blocks=38,
        axes_dim=[16, 56, 56], theta=10000, qkv_bias=True, guidance_embed=True
    )
    flux = Flux(flux_params).to(device).to(dtype)
    
    # Load Adapter
    adapter = VisualContextAdapter(
        in_channels=3,
        hidden_size=3072,
        downsample_factor=8
    ).to(device).to(dtype)
    
    if os.path.exists(args.adapter_path):
        adapter.load_state_dict(torch.load(args.adapter_path))
        print("Adapter loaded.")
        
    # 2. 预处理数据
    print("Processing input...")
    skeleton_img = load_image(args.skeleton_path, target_size=(1024, 1024))
    # Preprocess skeleton to tensor [1, 3, H, W]
    skeleton_tensor = torch.from_numpy(np.array(skeleton_img)).float() / 127.5 - 1.0
    skeleton_tensor = skeleton_tensor.permute(2, 0, 1).unsqueeze(0).to(device).to(dtype)
    
    # Adapter Forward
    with torch.no_grad():
        adapter_features = adapter(skeleton_tensor)
        
    # Prepare Flux inputs
    noise = get_noise(1, 1024, 1024, device, dtype, seed=42)
    inp = prepare(t5, clip, noise, args.prompt)
    
    # 3. 生成循环
    print("Generating...")
    # z = denoise(flux, **inp, adapter_features=adapter_features)
    
    # Decode
    # img = ae.decode(z)
    # img.save(args.output_path)
    
    print("Done (Mock run)")

if __name__ == "__main__":
    main()
