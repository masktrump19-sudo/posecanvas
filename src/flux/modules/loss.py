"""
骨架引导损失函数模块

本模块实现项目核心创新之一：骨架引导注意力损失 (Skeleton-Guided Attention Loss)

核心公式:
    L_skel = 1/(H×W) × Σ W(x,y) × |I_gen(x,y) - I_gt(x,y)|²
    
其中权重矩阵:
    W(x,y) = 1 + λ × M(x,y)
    
    - M(x,y): 骨架区域掩码（骨架/道具区域为1，背景为0）
    - λ: 骨架区域权重因子，默认值5.0（对骨架区域损失放大6倍）

损失组合:
    L_total = L_diffusion + α × L_skel
    
    - L_diffusion: 标准扩散模型损失（noise prediction MSE）
    - α: 骨架损失权重系数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class LossConfig:
    """损失函数配置"""
    # 骨架引导损失参数
    skeleton_lambda: float = 5.0  # λ参数，骨架区域权重因子
    skeleton_loss_weight: float = 0.1  # α参数，骨架损失在总损失中的权重
    
    # 扩散模型损失参数
    diffusion_loss_type: str = "mse"  # "mse" 或 "huber"
    huber_delta: float = 0.1
    
    # 感知损失参数（可选）
    use_perceptual_loss: bool = False
    perceptual_weight: float = 0.01
    perceptual_layers: Tuple[str, ...] = ("conv1_2", "conv2_2", "conv3_3")
    
    # 梯度惩罚（可选，用于平滑边界）
    use_gradient_penalty: bool = False
    gradient_penalty_weight: float = 0.001
    
    # 损失缩放（用于混合精度训练）
    loss_scale: float = 1.0


class SkeletonMaskGenerator(nn.Module):
    """
    骨架掩码生成器
    
    从骨架图生成权重掩码，用于计算骨架引导损失
    """
    
    def __init__(
        self,
        threshold: float = 0.1,
        dilation_kernel_size: int = 3,
        smooth_sigma: float = 1.0
    ):
        """
        Args:
            threshold: 骨架检测阈值（归一化到[-1,1]的图像）
            dilation_kernel_size: 膨胀核大小
            smooth_sigma: 高斯平滑的sigma值
        """
        super().__init__()
        self.threshold = threshold
        self.dilation_kernel_size = dilation_kernel_size
        self.smooth_sigma = smooth_sigma
        
        # 预计算膨胀卷积核
        if dilation_kernel_size > 0:
            kernel = torch.ones(1, 1, dilation_kernel_size, dilation_kernel_size)
            self.register_buffer("dilation_kernel", kernel)
        
    def forward(self, skeleton_map: torch.Tensor) -> torch.Tensor:
        """
        从骨架图生成掩码
        
        Args:
            skeleton_map: 骨架图 [B, 3, H, W]，范围[-1, 1]
            
        Returns:
            掩码 [B, 1, H, W]，范围[0, 1]
        """
        # 转换到[0, 1]范围
        skeleton_normalized = (skeleton_map + 1.0) / 2.0
        
        # 计算灰度图（简单平均）
        gray = skeleton_normalized.mean(dim=1, keepdim=True)
        
        # 阈值化：非黑色区域为骨架
        mask = (gray > self.threshold).float()
        
        # 膨胀操作（扩大骨架区域）
        if self.dilation_kernel_size > 0:
            padding = self.dilation_kernel_size // 2
            mask = F.conv2d(
                mask, 
                self.dilation_kernel.to(mask.device), 
                padding=padding
            )
            mask = (mask > 0).float()
        
        # 高斯平滑（可选，使边界更平滑）
        if self.smooth_sigma > 0:
            mask = self._gaussian_blur(mask, self.smooth_sigma)
        
        return mask
    
    def _gaussian_blur(
        self, 
        x: torch.Tensor, 
        sigma: float
    ) -> torch.Tensor:
        """高斯模糊"""
        kernel_size = int(4 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # 创建高斯核
        coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device)
        coords = coords - kernel_size // 2
        kernel_1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # 分离卷积
        kernel_h = kernel_1d.view(1, 1, -1, 1)
        kernel_w = kernel_1d.view(1, 1, 1, -1)
        
        padding = kernel_size // 2
        x = F.conv2d(x, kernel_h, padding=(padding, 0))
        x = F.conv2d(x, kernel_w, padding=(0, padding))
        
        return x


class SkeletonGuidedLoss(nn.Module):
    """
    骨架引导注意力损失 (Skeleton-Guided Attention Loss)
    
    核心创新：通过空间加权使模型在训练时更关注骨架/道具区域，
    从而提高生成图像中人物姿态和道具位置的准确性。
    
    公式:
        L_skel = mean(W × |pred - target|²)
        W(x,y) = 1 + λ × M(x,y)
    
    其中：
        - pred: 模型预测（噪声或图像）
        - target: 目标值
        - M: 骨架区域掩码
        - λ: 权重因子（默认5.0）
    """
    
    def __init__(
        self,
        lambda_weight: float = 5.0,
        reduction: str = "mean",
        mask_generator: Optional[SkeletonMaskGenerator] = None
    ):
        """
        Args:
            lambda_weight: λ参数，骨架区域额外权重
            reduction: 损失归约方式 ("mean", "sum", "none")
            mask_generator: 掩码生成器（如果为None则需要外部提供掩码）
        """
        super().__init__()
        self.lambda_weight = lambda_weight
        self.reduction = reduction
        self.mask_generator = mask_generator or SkeletonMaskGenerator()
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        skeleton_map: Optional[torch.Tensor] = None,
        skeleton_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算骨架引导损失
        
        Args:
            pred: 预测值 [B, C, H, W]
            target: 目标值 [B, C, H, W]
            skeleton_map: 骨架图 [B, 3, H, W]（用于生成掩码）
            skeleton_mask: 预先计算的掩码 [B, 1, H, W]（可选）
            
        Returns:
            损失值
        """
        # 获取或生成掩码
        if skeleton_mask is None:
            if skeleton_map is None:
                raise ValueError("必须提供skeleton_map或skeleton_mask之一")
            skeleton_mask = self.mask_generator(skeleton_map)
        
        # 调整掩码大小以匹配预测
        if skeleton_mask.shape[-2:] != pred.shape[-2:]:
            skeleton_mask = F.interpolate(
                skeleton_mask,
                size=pred.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
        
        # 计算权重矩阵: W = 1 + λ × M
        weights = 1.0 + self.lambda_weight * skeleton_mask
        
        # 计算加权MSE损失
        squared_error = (pred - target) ** 2
        weighted_error = weights * squared_error
        
        # 归约
        if self.reduction == "mean":
            loss = weighted_error.mean()
        elif self.reduction == "sum":
            loss = weighted_error.sum()
        else:
            loss = weighted_error
            
        return loss
    
    def compute_weight_stats(
        self,
        skeleton_mask: torch.Tensor
    ) -> Dict[str, float]:
        """计算权重统计信息（用于监控）"""
        weights = 1.0 + self.lambda_weight * skeleton_mask
        return {
            "weight_mean": weights.mean().item(),
            "weight_max": weights.max().item(),
            "weight_min": weights.min().item(),
            "skeleton_coverage": skeleton_mask.mean().item(),
        }


class DiffusionLoss(nn.Module):
    """
    标准扩散模型损失
    
    支持MSE和Huber损失
    """
    
    def __init__(
        self,
        loss_type: str = "mse",
        huber_delta: float = 0.1,
        reduction: str = "mean"
    ):
        super().__init__()
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.reduction = reduction
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算扩散损失
        
        Args:
            pred: 预测噪声 [B, C, H, W]
            target: 目标噪声 [B, C, H, W]
            
        Returns:
            损失值
        """
        if self.loss_type == "mse":
            loss = F.mse_loss(pred, target, reduction=self.reduction)
        elif self.loss_type == "huber":
            loss = F.huber_loss(
                pred, target, 
                delta=self.huber_delta, 
                reduction=self.reduction
            )
        elif self.loss_type == "l1":
            loss = F.l1_loss(pred, target, reduction=self.reduction)
        else:
            raise ValueError(f"未知损失类型: {self.loss_type}")
            
        return loss


class PerceptualLoss(nn.Module):
    """
    感知损失（基于VGG特征）
    
    可选组件，用于提高生成图像的视觉质量
    """
    
    def __init__(
        self,
        layers: Tuple[str, ...] = ("conv1_2", "conv2_2", "conv3_3"),
        weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        self.layers = layers
        self.weights = weights or {layer: 1.0 for layer in layers}
        
        # 延迟加载VGG（避免在不使用时占用显存）
        self._vgg = None
        
    def _load_vgg(self, device: torch.device):
        """延迟加载VGG模型"""
        if self._vgg is None:
            try:
                from torchvision.models import vgg16, VGG16_Weights
                vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
            except ImportError:
                # 兼容旧版本torchvision
                from torchvision.models import vgg16
                vgg = vgg16(pretrained=True).features
            
            vgg = vgg.eval()
            for param in vgg.parameters():
                param.requires_grad = False
            self._vgg = vgg.to(device)
            
            # VGG层名到索引的映射
            self._layer_map = {
                "conv1_1": 0, "conv1_2": 2,
                "conv2_1": 5, "conv2_2": 7,
                "conv3_1": 10, "conv3_2": 12, "conv3_3": 14,
                "conv4_1": 17, "conv4_2": 19, "conv4_3": 21,
                "conv5_1": 24, "conv5_2": 26, "conv5_3": 28,
            }
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算感知损失
        
        Args:
            pred: 预测图像 [B, 3, H, W]，范围[-1, 1]
            target: 目标图像 [B, 3, H, W]，范围[-1, 1]
            
        Returns:
            感知损失
        """
        self._load_vgg(pred.device)
        
        # 归一化到ImageNet范围
        mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pred.device).view(1, 3, 1, 1)
        
        pred_norm = ((pred + 1) / 2 - mean) / std
        target_norm = ((target + 1) / 2 - mean) / std
        
        # 提取特征并计算损失
        total_loss = 0.0
        pred_features = pred_norm
        target_features = target_norm
        
        for name, layer in self._vgg._modules.items():
            pred_features = layer(pred_features)
            target_features = layer(target_features)
            
            layer_name = f"conv{int(name)//3 + 1}_{int(name)%3 + 1}"
            if layer_name in self.layers:
                weight = self.weights.get(layer_name, 1.0)
                total_loss += weight * F.mse_loss(pred_features, target_features)
        
        return total_loss


class GradientPenalty(nn.Module):
    """
    梯度惩罚
    
    用于平滑骨架边界处的生成结果
    """
    
    def __init__(self):
        super().__init__()
        
        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))
        
    def forward(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算梯度惩罚
        
        Args:
            image: 图像 [B, C, H, W]
            mask: 边界掩码 [B, 1, H, W]（可选，用于聚焦边界区域）
            
        Returns:
            梯度惩罚损失
        """
        # 对每个通道计算梯度
        grad_x = F.conv2d(image, self.sobel_x.repeat(image.size(1), 1, 1, 1), 
                          padding=1, groups=image.size(1))
        grad_y = F.conv2d(image, self.sobel_y.repeat(image.size(1), 1, 1, 1),
                          padding=1, groups=image.size(1))
        
        # 梯度幅值
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
        # 如果提供掩码，只计算掩码区域的梯度
        if mask is not None:
            gradient_magnitude = gradient_magnitude * mask
            
        return gradient_magnitude.mean()


class CombinedLoss(nn.Module):
    """
    组合损失函数
    
    整合所有损失组件：
    1. 扩散损失（必需）
    2. 骨架引导损失（必需）
    3. 感知损失（可选）
    4. 梯度惩罚（可选）
    
    公式:
        L_total = L_diffusion + α × L_skel + β × L_perceptual + γ × L_gradient
    """
    
    def __init__(self, config: Optional[LossConfig] = None):
        """
        Args:
            config: 损失函数配置
        """
        super().__init__()
        self.config = config or LossConfig()
        
        # 核心损失组件
        self.diffusion_loss = DiffusionLoss(
            loss_type=self.config.diffusion_loss_type,
            huber_delta=self.config.huber_delta
        )
        
        self.skeleton_loss = SkeletonGuidedLoss(
            lambda_weight=self.config.skeleton_lambda
        )
        
        # 可选组件
        self.perceptual_loss = None
        if self.config.use_perceptual_loss:
            self.perceptual_loss = PerceptualLoss(
                layers=self.config.perceptual_layers
            )
            
        self.gradient_penalty = None
        if self.config.use_gradient_penalty:
            self.gradient_penalty = GradientPenalty()
            
    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        skeleton_map: Optional[torch.Tensor] = None,
        skeleton_mask: Optional[torch.Tensor] = None,
        image_pred: Optional[torch.Tensor] = None,
        image_target: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算组合损失
        
        Args:
            noise_pred: 预测噪声 [B, C, H, W]
            noise_target: 目标噪声 [B, C, H, W]
            skeleton_map: 骨架图 [B, 3, H, W]
            skeleton_mask: 骨架掩码 [B, 1, H, W]
            image_pred: 预测图像（用于感知损失）
            image_target: 目标图像（用于感知损失）
            
        Returns:
            损失字典 {"total": ..., "diffusion": ..., "skeleton": ..., ...}
        """
        losses = {}
        
        # 1. 扩散损失
        loss_diffusion = self.diffusion_loss(noise_pred, noise_target)
        losses["diffusion"] = loss_diffusion
        
        # 2. 骨架引导损失
        loss_skeleton = self.skeleton_loss(
            noise_pred, noise_target,
            skeleton_map=skeleton_map,
            skeleton_mask=skeleton_mask
        )
        losses["skeleton"] = loss_skeleton
        
        # 3. 感知损失（可选）
        loss_perceptual = torch.tensor(0.0, device=noise_pred.device)
        if self.perceptual_loss is not None and image_pred is not None:
            loss_perceptual = self.perceptual_loss(image_pred, image_target)
            losses["perceptual"] = loss_perceptual
        
        # 4. 梯度惩罚（可选）
        loss_gradient = torch.tensor(0.0, device=noise_pred.device)
        if self.gradient_penalty is not None and image_pred is not None:
            loss_gradient = self.gradient_penalty(image_pred, skeleton_mask)
            losses["gradient"] = loss_gradient
        
        # 总损失
        total_loss = (
            loss_diffusion +
            self.config.skeleton_loss_weight * loss_skeleton +
            self.config.perceptual_weight * loss_perceptual +
            self.config.gradient_penalty_weight * loss_gradient
        )
        
        # 损失缩放
        total_loss = total_loss * self.config.loss_scale
        losses["total"] = total_loss
        
        return losses


class FlowMatchingLoss(nn.Module):
    """
    Flow Matching损失（用于Flux.1架构）
    
    Flux.1使用Rectified Flow而非传统扩散，损失计算略有不同
    """
    
    def __init__(
        self,
        config: Optional[LossConfig] = None
    ):
        super().__init__()
        self.config = config or LossConfig()
        
        self.skeleton_loss = SkeletonGuidedLoss(
            lambda_weight=self.config.skeleton_lambda
        )
        
    def forward(
        self,
        velocity_pred: torch.Tensor,
        velocity_target: torch.Tensor,
        skeleton_map: Optional[torch.Tensor] = None,
        skeleton_mask: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算Flow Matching损失
        
        Args:
            velocity_pred: 预测速度场 [B, C, H, W]
            velocity_target: 目标速度场 [B, C, H, W]
            skeleton_map: 骨架图
            skeleton_mask: 骨架掩码
            timesteps: 时间步（可用于时间加权）
            
        Returns:
            损失字典
        """
        losses = {}
        
        # 基础Flow Matching损失
        loss_flow = F.mse_loss(velocity_pred, velocity_target)
        losses["flow"] = loss_flow
        
        # 骨架引导损失
        loss_skeleton = self.skeleton_loss(
            velocity_pred, velocity_target,
            skeleton_map=skeleton_map,
            skeleton_mask=skeleton_mask
        )
        losses["skeleton"] = loss_skeleton
        
        # 总损失
        total_loss = loss_flow + self.config.skeleton_loss_weight * loss_skeleton
        losses["total"] = total_loss
        
        return losses


# ==================== 工具函数 ====================

def create_loss_function(
    loss_type: str = "combined",
    **kwargs
) -> nn.Module:
    """
    创建损失函数的工厂函数
    
    Args:
        loss_type: 损失类型 ("combined", "skeleton", "diffusion", "flow_matching")
        **kwargs: 配置参数
        
    Returns:
        损失函数实例
    """
    config = LossConfig(**kwargs)
    
    if loss_type == "combined":
        return CombinedLoss(config)
    elif loss_type == "skeleton":
        return SkeletonGuidedLoss(lambda_weight=config.skeleton_lambda)
    elif loss_type == "diffusion":
        return DiffusionLoss(
            loss_type=config.diffusion_loss_type,
            huber_delta=config.huber_delta
        )
    elif loss_type == "flow_matching":
        return FlowMatchingLoss(config)
    else:
        raise ValueError(f"未知损失类型: {loss_type}")


def compute_skeleton_coverage(
    skeleton_map: torch.Tensor,
    threshold: float = 0.1
) -> float:
    """
    计算骨架图覆盖率（用于数据分析）
    
    Args:
        skeleton_map: 骨架图 [B, 3, H, W]
        threshold: 检测阈值
        
    Returns:
        覆盖率 (0-1)
    """
    skeleton_normalized = (skeleton_map + 1.0) / 2.0
    gray = skeleton_normalized.mean(dim=1)
    mask = (gray > threshold).float()
    return mask.mean().item()


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("测试损失函数模块...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建测试数据
    batch_size = 2
    channels = 16  # Flux.1潜空间通道数
    height, width = 64, 64  # 潜空间尺寸
    
    noise_pred = torch.randn(batch_size, channels, height, width, device=device)
    noise_target = torch.randn(batch_size, channels, height, width, device=device)
    skeleton_map = torch.randn(batch_size, 3, height * 8, width * 8, device=device)  # 骨架图原始分辨率
    
    print(f"\n输入形状:")
    print(f"  noise_pred: {noise_pred.shape}")
    print(f"  noise_target: {noise_target.shape}")
    print(f"  skeleton_map: {skeleton_map.shape}")
    
    print("\n1. 测试SkeletonMaskGenerator...")
    mask_gen = SkeletonMaskGenerator()
    mask = mask_gen(skeleton_map)
    print(f"   掩码形状: {mask.shape}")
    print(f"   掩码范围: [{mask.min():.4f}, {mask.max():.4f}]")
    
    print("\n2. 测试SkeletonGuidedLoss...")
    skel_loss = SkeletonGuidedLoss(lambda_weight=5.0)
    loss_skel = skel_loss(noise_pred, noise_target, skeleton_map=skeleton_map)
    print(f"   骨架引导损失: {loss_skel.item():.6f}")
    
    # 测试权重统计
    mask_downsampled = F.interpolate(mask, size=(height, width), mode="bilinear")
    stats = skel_loss.compute_weight_stats(mask_downsampled)
    print(f"   权重统计: {stats}")
    
    print("\n3. 测试DiffusionLoss...")
    diff_loss = DiffusionLoss(loss_type="mse")
    loss_diff = diff_loss(noise_pred, noise_target)
    print(f"   扩散损失 (MSE): {loss_diff.item():.6f}")
    
    diff_loss_huber = DiffusionLoss(loss_type="huber", huber_delta=0.1)
    loss_huber = diff_loss_huber(noise_pred, noise_target)
    print(f"   扩散损失 (Huber): {loss_huber.item():.6f}")
    
    print("\n4. 测试CombinedLoss...")
    config = LossConfig(
        skeleton_lambda=5.0,
        skeleton_loss_weight=0.1,
        use_perceptual_loss=False,
        use_gradient_penalty=False
    )
    combined_loss = CombinedLoss(config)
    losses = combined_loss(
        noise_pred, noise_target,
        skeleton_map=skeleton_map
    )
    print(f"   损失组成:")
    for name, value in losses.items():
        print(f"     {name}: {value.item():.6f}")
    
    print("\n5. 测试FlowMatchingLoss...")
    flow_loss = FlowMatchingLoss(config)
    velocity_pred = torch.randn(batch_size, channels, height, width, device=device)
    velocity_target = torch.randn(batch_size, channels, height, width, device=device)
    losses_flow = flow_loss(
        velocity_pred, velocity_target,
        skeleton_map=skeleton_map
    )
    print(f"   Flow Matching损失:")
    for name, value in losses_flow.items():
        print(f"     {name}: {value.item():.6f}")
    
    print("\n6. 测试工厂函数...")
    loss_fn = create_loss_function("combined", skeleton_lambda=5.0)
    print(f"   创建的损失函数类型: {type(loss_fn).__name__}")
    
    print("\n7. 测试梯度计算...")
    noise_pred.requires_grad = True
    losses = combined_loss(noise_pred, noise_target, skeleton_map=skeleton_map)
    losses["total"].backward()
    print(f"   梯度范数: {noise_pred.grad.norm().item():.6f}")
    
    print("\n✓ 所有测试通过!")