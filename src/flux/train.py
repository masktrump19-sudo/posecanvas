"""
Flux Model 训练脚本

主要功能:
1. 加载和配置 Visual Context Adapter
2. 加载预训练 Flux.1 模型
3. 准备骨架数据集
4. 执行训练循环 (Skeleton-Guided Training)

Supports:
- Skeleton-Guided Attention Loss
- Visual Context Adapter Training
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flux.model import Flux, FluxParams
from flux.modules.adapter import VisualContextAdapter
from flux.modules.loss import SkeletonGuidedLoss
from flux.data.dataset import create_dataloader

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_checkpoint(model, adapter, optimizer, epoch, step, path):
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(), # 若只训练adapter可不存
        'adapter_state_dict': adapter.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved to {path}")

def main():
    parser = argparse.ArgumentParser(description="Train Flux with Visual Context Adapter")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to config file")
    args = parser.parse_args()

    # 1. 加载配置
    config = load_config(args.config)
    device = torch.device(config.get("device", "cuda"))
    
    # 2. 初始化模型
    print("Initializing models...")
    # 注意: 这里简化了Flux参数初始化，实际应从预训练权重配置加载
    # 假设有一个 load_flux_model 辅助函数或直接使用 HuggingFace
    # 此处仅作代码结构演示
    flux_params = FluxParams(
        in_channels=64, # 假设 latent channels
        out_channels=64,
        vec_in_dim=768,
        context_in_dim=4096,
        hidden_size=config["model"]["adapter"]["hidden_size"],
        mlp_ratio=4.0,
        num_heads=24,
        depth=19,
        depth_single_blocks=38,
        axes_dim=[16, 56, 56],
        theta=10000,
        qkv_bias=True,
        guidance_embed=True
    )
    model = Flux(flux_params).to(device)
    
    # 初始化 Adapter
    adapter = VisualContextAdapter(
        in_channels=config["model"]["adapter"]["in_channels"],
        hidden_size=config["model"]["adapter"]["hidden_size"],
        downsample_factor=config["model"]["adapter"]["downsample_factor"]
    ).to(device)
    
    # 3. 准备数据
    print("Loading data...")
    # 假设使用伪数据或根据Dataset模块接口调用
    # dataloader = create_dataloader(...) 
    # 这里为了能运行，暂时留空或mock
    dataloader = [] 

    # 4. 定义损失函数
    loss_fn = SkeletonGuidedLoss(
        lambda_weight=config["model"]["loss"]["skeleton_lambda"]
    ).to(device)
    
    # 5. 优化器
    # 仅训练 Adapter
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=config["train"]["weight_decay"]
    )
    
    # 6. 训练循环
    num_epochs = config["train"]["num_epochs"]
    global_step = 0
    
    model.train() # 如果只finetune adapter，可以将model.eval()并冻结参数
    adapter.train()
    
    # Freeze Flux model
    for param in model.parameters():
        param.requires_grad = False
        
    print(f"Start training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # batch: {image, skeleton, text_emb, ...}
            # 此处需根据实际Dataset输出解包
            
            # skeleton_img = batch['skeleton'].to(device)
            # target_img = batch['image'].to(device)
            # ...
            
            optimizer.zero_grad()
            
            # Forward Adapter
            # adapter_features = adapter(skeleton_img)
            
            # Forward Flux with Adapter Features
            # note: 需要修改Flux forward接口以接受adapter_features
            # pred = model(..., adapter_features=adapter_features)
            
            # pred = torch.randn_like(target_img) # Mock prediction
            
            # Calculate Loss
            # loss = loss_fn(pred, target_img, skeleton_map=skeleton_img)
            
            # Backward
            # loss.backward()
            # optimizer.step()
            
            global_step += 1
            # progress_bar.set_postfix(loss=loss.item())
            
            if global_step % config["train"]["save_interval"] == 0:
                save_path = os.path.join(config["output_dir"], f"checkpoint_step_{global_step}.pt")
                # save_checkpoint(...)
                
    print("Training finished.")

if __name__ == "__main__":
    main()
