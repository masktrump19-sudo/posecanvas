"""
AnimePoseDataset: 动漫姿势数据集加载器

用于加载（目标图像，密集文本描述，视觉骨架图）三元组数据
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class AnimePoseDataset(Dataset):
    """
    动漫姿势三元组数据集
    
    数据格式 (JSONL):
    {
        "target_image_path": "path/to/image.png",
        "skeleton_path": "path/to/skeleton.png", 
        "caption": "A majestic anime illustration of...",
        "scene_type": "ancient_fantasy",  # optional
        "has_prop": true  # optional
    }
    
    Args:
        data_path: JSONL数据文件路径或包含数据文件的目录
        image_size: 输出图像尺寸 (默认1024)
        tokenizer: 文本tokenizer (可选，用于预tokenize)
        max_token_length: 最大token长度 (默认512)
        augment: 是否启用数据增强
        return_paths: 是否返回文件路径信息
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        image_size: int = 1024,
        tokenizer=None,
        max_token_length: int = 512,
        augment: bool = False,
        return_paths: bool = False,
    ):
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        self.augment = augment
        self.return_paths = return_paths
        
        # 加载数据
        self.data = self._load_data()
        
        # 定义图像变换
        # 骨架图使用NEAREST插值保持线条清晰度
        self.skeleton_transform = transforms.Compose([
            transforms.Resize(
                image_size, 
                interpolation=InterpolationMode.NEAREST
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
        
        # 目标图像归一化到[-1, 1]范围 (Flux标准)
        self.image_transform = transforms.Compose([
            transforms.Resize(
                image_size,
                interpolation=InterpolationMode.LANCZOS
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        
        # 数据增强变换
        if self.augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.1, 
                    contrast=0.1, 
                    saturation=0.1
                ),
            ])
        else:
            self.augment_transform = None
    
    def _load_data(self) -> List[Dict]:
        """加载JSONL格式的数据"""
        data = []
        
        if self.data_path.is_file():
            # 单个JSONL文件
            data = self._load_jsonl(self.data_path)
        elif self.data_path.is_dir():
            # 目录中的所有JSONL文件
            for jsonl_file in self.data_path.glob("*.jsonl"):
                data.extend(self._load_jsonl(jsonl_file))
            # 也支持JSON文件
            for json_file in self.data_path.glob("*.json"):
                with open(json_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        data.extend(loaded)
                    else:
                        data.append(loaded)
        else:
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        
        # 验证数据完整性
        valid_data = []
        for item in data:
            if self._validate_item(item):
                valid_data.append(item)
            else:
                print(f"Warning: Skipping invalid item: {item.get('target_image_path', 'unknown')}")
        
        print(f"Loaded {len(valid_data)} valid samples from {len(data)} total entries")
        return valid_data
    
    def _load_jsonl(self, filepath: Path) -> List[Dict]:
        """加载单个JSONL文件"""
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line in {filepath}: {e}")
        return data
    
    def _validate_item(self, item: Dict) -> bool:
        """验证数据项的完整性"""
        required_keys = ["target_image_path", "skeleton_path", "caption"]
        for key in required_keys:
            if key not in item:
                return False
        
        # 检查文件是否存在
        base_dir = self.data_path.parent if self.data_path.is_file() else self.data_path
        
        img_path = Path(item["target_image_path"])
        if not img_path.is_absolute():
            img_path = base_dir / img_path
        
        skel_path = Path(item["skeleton_path"])
        if not skel_path.is_absolute():
            skel_path = base_dir / skel_path
        
        return img_path.exists() and skel_path.exists()
    
    def _resolve_path(self, path_str: str) -> Path:
        """解析相对/绝对路径"""
        path = Path(path_str)
        if path.is_absolute():
            return path
        base_dir = self.data_path.parent if self.data_path.is_file() else self.data_path
        return base_dir / path
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # 加载图像
        img_path = self._resolve_path(item["target_image_path"])
        skel_path = self._resolve_path(item["skeleton_path"])
        
        image = Image.open(img_path).convert("RGB")
        skeleton = Image.open(skel_path).convert("RGB")
        
        # 数据增强 (同步应用到图像和骨架)
        if self.augment and self.augment_transform is not None:
            # 使用相同的随机种子确保同步
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            image = self.augment_transform(image)
            torch.manual_seed(seed)
            skeleton = self.augment_transform(skeleton)
        
        # 应用变换
        pixel_values = self.image_transform(image)
        conditioning = self.skeleton_transform(skeleton)
        
        # 构建返回字典
        result = {
            "pixel_values": pixel_values,           # [3, H, W], 范围[-1, 1]
            "conditioning": conditioning,            # [3, H, W], 范围[0, 1]
            "prompt": item["caption"],
        }
        
        # 可选: 预tokenize文本
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                item["caption"],
                max_length=self.max_token_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            result["input_ids"] = tokens["input_ids"].squeeze(0)
            result["attention_mask"] = tokens["attention_mask"].squeeze(0)
        
        # 可选: 返回元数据
        if self.return_paths:
            result["image_path"] = str(img_path)
            result["skeleton_path"] = str(skel_path)
            result["scene_type"] = item.get("scene_type", "unknown")
            result["has_prop"] = item.get("has_prop", False)
        
        return result


class LatentAnimePoseDataset(AnimePoseDataset):
    """
    预编码的潜空间数据集
    
    用于加速训练，预先将图像编码为潜空间表示
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        latent_dir: Union[str, Path],
        image_size: int = 1024,
        tokenizer=None,
        max_token_length: int = 512,
    ):
        super().__init__(
            data_path=data_path,
            image_size=image_size,
            tokenizer=tokenizer,
            max_token_length=max_token_length,
            augment=False,
            return_paths=True,
        )
        self.latent_dir = Path(latent_dir)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = super().__getitem__(idx)
        
        # 尝试加载预编码的潜空间
        img_path = Path(result["image_path"])
        latent_path = self.latent_dir / f"{img_path.stem}_latent.pt"
        
        if latent_path.exists():
            result["latents"] = torch.load(latent_path)
            # 删除原始像素值以节省内存
            del result["pixel_values"]
        
        return result


def create_dataloader(
    data_path: Union[str, Path],
    batch_size: int = 2,
    image_size: int = 1024,
    tokenizer=None,
    max_token_length: int = 512,
    augment: bool = False,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    创建数据加载器的便捷函数
    
    Args:
        data_path: 数据文件路径
        batch_size: 批量大小
        image_size: 图像尺寸
        tokenizer: 文本tokenizer
        max_token_length: 最大token长度
        augment: 是否启用数据增强
        shuffle: 是否打乱数据
        num_workers: 数据加载工作线程数
        pin_memory: 是否固定内存
        drop_last: 是否丢弃最后不完整的batch
    
    Returns:
        DataLoader实例
    """
    dataset = AnimePoseDataset(
        data_path=data_path,
        image_size=image_size,
        tokenizer=tokenizer,
        max_token_length=max_token_length,
        augment=augment,
        return_paths=False,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    
    return dataloader


def collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """
    自定义collate函数，处理变长文本
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    conditioning = torch.stack([item["conditioning"] for item in batch])
    prompts = [item["prompt"] for item in batch]
    
    result = {
        "pixel_values": pixel_values,
        "conditioning": conditioning,
        "prompt": prompts,
    }
    
    # 如果有tokenized的输入
    if "input_ids" in batch[0]:
        result["input_ids"] = torch.stack([item["input_ids"] for item in batch])
        result["attention_mask"] = torch.stack([item["attention_mask"] for item in batch])
    
    return result


# ============== 数据集创建工具函数 ==============

def create_triplet_entry(
    target_image_path: str,
    skeleton_path: str,
    caption: str,
    scene_type: str = "unknown",
    has_prop: bool = False,
) -> Dict:
    """创建单条三元组数据"""
    return {
        "target_image_path": target_image_path,
        "skeleton_path": skeleton_path,
        "caption": caption,
        "scene_type": scene_type,
        "has_prop": has_prop,
    }


def save_dataset_jsonl(
    entries: List[Dict],
    output_path: Union[str, Path],
):
    """保存数据集为JSONL格式"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(entries)} entries to {output_path}")


if __name__ == "__main__":
    # 测试数据集
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=1024)
    args = parser.parse_args()
    
    dataset = AnimePoseDataset(
        data_path=args.data_path,
        image_size=args.image_size,
        return_paths=True,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Pixel values shape: {sample['pixel_values'].shape}")
        print(f"Conditioning shape: {sample['conditioning'].shape}")
        print(f"Prompt: {sample['prompt'][:100]}...")