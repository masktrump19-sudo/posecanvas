"""
骨架预处理与道具感知骨架图生成模块

本模块实现:
1. SkeletonPreprocessor: 骨架图预处理（标准化、增强、格式转换）
2. PropAwareSkeletonGenerator: 道具感知视觉骨架图生成
3. KeypointProcessor: 关键点检测与处理

道具感知视觉骨架图 (Prop-Aware Visual Skeleton Map):
- 18个标准人体关键点 (COCO格式)
- 道具轮廓以青色 (Cyan, RGB: 0, 255, 255) 绘制
- 骨架连接线以白色绘制，关键点以不同颜色标识
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import json


# ==================== 常量定义 ====================

# COCO 18关键点定义
COCO_KEYPOINTS = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
    "neck",           # 17 (可选，某些格式可能没有)
]

# 骨架连接定义 (关键点索引对)
SKELETON_CONNECTIONS = [
    # 头部
    (0, 1), (0, 2), (1, 3), (2, 4),
    # 躯干
    (0, 17), (17, 5), (17, 6), (5, 11), (6, 12), (11, 12),
    # 左臂
    (5, 7), (7, 9),
    # 右臂
    (6, 8), (8, 10),
    # 左腿
    (11, 13), (13, 15),
    # 右腿
    (12, 14), (14, 16),
]

# 备选骨架连接（无neck关键点时）
SKELETON_CONNECTIONS_NO_NECK = [
    # 头部
    (0, 1), (0, 2), (1, 3), (2, 4),
    # 躯干 (直接连接肩膀)
    (0, 5), (0, 6), (5, 6), (5, 11), (6, 12), (11, 12),
    # 左臂
    (5, 7), (7, 9),
    # 右臂
    (6, 8), (8, 10),
    # 左腿
    (11, 13), (13, 15),
    # 右腿
    (12, 14), (14, 16),
]

# 关键点颜色 (BGR格式，用于可视化)
KEYPOINT_COLORS = {
    "head": (255, 0, 0),      # 蓝色 - 头部关键点
    "torso": (0, 255, 0),     # 绿色 - 躯干关键点
    "left_arm": (255, 255, 0), # 青色 - 左臂
    "right_arm": (0, 255, 255), # 黄色 - 右臂
    "left_leg": (255, 0, 255),  # 品红 - 左腿
    "right_leg": (128, 0, 255), # 橙色 - 右腿
}

# 道具轮廓颜色 (Cyan)
PROP_COLOR = (255, 255, 0)  # BGR格式的青色
PROP_COLOR_RGB = (0, 255, 255)  # RGB格式的青色

# 骨架线颜色
SKELETON_LINE_COLOR = (255, 255, 255)  # 白色


@dataclass
class SkeletonConfig:
    """骨架处理配置"""
    # 图像尺寸
    target_size: Tuple[int, int] = (1024, 1024)
    
    # 绘制参数
    keypoint_radius: int = 4
    skeleton_thickness: int = 2
    prop_thickness: int = 2
    
    # 关键点置信度阈值
    confidence_threshold: float = 0.3
    
    # 是否使用neck关键点
    use_neck: bool = True
    
    # 数据增强
    enable_augmentation: bool = True
    scale_range: Tuple[float, float] = (0.8, 1.2)
    rotation_range: Tuple[float, float] = (-15, 15)
    translation_range: Tuple[float, float] = (-0.1, 0.1)
    flip_probability: float = 0.5
    
    # 道具检测
    enable_prop_detection: bool = True
    prop_confidence_threshold: float = 0.5


class KeypointProcessor:
    """关键点检测与处理类"""
    
    def __init__(self, config: Optional[SkeletonConfig] = None):
        self.config = config or SkeletonConfig()
        
    def parse_keypoints(
        self,
        keypoints: Union[np.ndarray, List, Dict],
        format: str = "coco"
    ) -> np.ndarray:
        """
        解析不同格式的关键点数据
        
        Args:
            keypoints: 关键点数据
            format: 数据格式 ("coco", "openpose", "mediapipe")
            
        Returns:
            标准化的关键点数组 [N, 3] (x, y, confidence)
        """
        if isinstance(keypoints, dict):
            # 字典格式 {"keypoints": [...]}
            if "keypoints" in keypoints:
                keypoints = keypoints["keypoints"]
        
        keypoints = np.array(keypoints, dtype=np.float32)
        
        # 根据形状推断格式
        if keypoints.ndim == 1:
            # 展平格式 [x1, y1, c1, x2, y2, c2, ...]
            n_keypoints = len(keypoints) // 3
            keypoints = keypoints.reshape(n_keypoints, 3)
        elif keypoints.ndim == 2:
            if keypoints.shape[1] == 2:
                # 仅坐标格式 [N, 2]
                conf = np.ones((keypoints.shape[0], 1), dtype=np.float32)
                keypoints = np.concatenate([keypoints, conf], axis=1)
        
        # 确保至少有17个关键点
        if keypoints.shape[0] < 17:
            padding = np.zeros((17 - keypoints.shape[0], 3), dtype=np.float32)
            keypoints = np.concatenate([keypoints, padding], axis=0)
        
        # 添加neck关键点（如果不存在）
        if keypoints.shape[0] == 17 and self.config.use_neck:
            # neck = (left_shoulder + right_shoulder) / 2
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            if left_shoulder[2] > 0 and right_shoulder[2] > 0:
                neck = (left_shoulder + right_shoulder) / 2
            else:
                neck = np.array([0, 0, 0], dtype=np.float32)
            keypoints = np.concatenate([keypoints, neck.reshape(1, 3)], axis=0)
            
        return keypoints
    
    def normalize_keypoints(
        self,
        keypoints: np.ndarray,
        image_size: Tuple[int, int],
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        标准化关键点坐标到目标尺寸
        
        Args:
            keypoints: 关键点数组 [N, 3]
            image_size: 原始图像尺寸 (width, height)
            target_size: 目标尺寸 (width, height)
            
        Returns:
            标准化后的关键点数组
        """
        target_size = target_size or self.config.target_size
        
        kp_normalized = keypoints.copy()
        
        # 缩放坐标
        scale_x = target_size[0] / image_size[0]
        scale_y = target_size[1] / image_size[1]
        
        kp_normalized[:, 0] *= scale_x
        kp_normalized[:, 1] *= scale_y
        
        return kp_normalized
    
    def filter_by_confidence(
        self,
        keypoints: np.ndarray,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """根据置信度过滤关键点"""
        threshold = threshold or self.config.confidence_threshold
        kp_filtered = keypoints.copy()
        mask = kp_filtered[:, 2] < threshold
        kp_filtered[mask, :2] = 0  # 低置信度关键点坐标置0
        return kp_filtered
    
    def augment_keypoints(
        self,
        keypoints: np.ndarray,
        image_size: Tuple[int, int],
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        数据增强：缩放、旋转、平移、翻转
        
        Args:
            keypoints: 关键点数组 [N, 3]
            image_size: 图像尺寸 (width, height)
            seed: 随机种子
            
        Returns:
            增强后的关键点和变换参数
        """
        if not self.config.enable_augmentation:
            return keypoints, {}
        
        if seed is not None:
            np.random.seed(seed)
        
        kp_aug = keypoints.copy()
        params = {}
        
        width, height = image_size
        center_x, center_y = width / 2, height / 2
        
        # 随机缩放
        scale = np.random.uniform(*self.config.scale_range)
        params["scale"] = scale
        
        # 随机旋转
        angle = np.random.uniform(*self.config.rotation_range)
        params["rotation"] = angle
        
        # 随机平移
        tx = np.random.uniform(*self.config.translation_range) * width
        ty = np.random.uniform(*self.config.translation_range) * height
        params["translation"] = (tx, ty)
        
        # 随机翻转
        flip = np.random.random() < self.config.flip_probability
        params["flip"] = flip
        
        # 应用变换
        for i in range(len(kp_aug)):
            if kp_aug[i, 2] > 0:  # 仅处理有效关键点
                x, y = kp_aug[i, :2]
                
                # 缩放（相对于中心）
                x = (x - center_x) * scale + center_x
                y = (y - center_y) * scale + center_y
                
                # 旋转（相对于中心）
                rad = np.radians(angle)
                cos_a, sin_a = np.cos(rad), np.sin(rad)
                x_new = cos_a * (x - center_x) - sin_a * (y - center_y) + center_x
                y_new = sin_a * (x - center_x) + cos_a * (y - center_y) + center_y
                x, y = x_new, y_new
                
                # 平移
                x += tx
                y += ty
                
                # 翻转
                if flip:
                    x = width - x
                
                kp_aug[i, :2] = [x, y]
        
        # 如果翻转，交换左右关键点
        if flip:
            kp_aug = self._swap_left_right(kp_aug)
        
        return kp_aug, params
    
    def _swap_left_right(self, keypoints: np.ndarray) -> np.ndarray:
        """翻转时交换左右关键点"""
        kp_swapped = keypoints.copy()
        
        # 左右眼
        kp_swapped[1], kp_swapped[2] = keypoints[2].copy(), keypoints[1].copy()
        # 左右耳
        kp_swapped[3], kp_swapped[4] = keypoints[4].copy(), keypoints[3].copy()
        # 左右肩
        kp_swapped[5], kp_swapped[6] = keypoints[6].copy(), keypoints[5].copy()
        # 左右肘
        kp_swapped[7], kp_swapped[8] = keypoints[8].copy(), keypoints[7].copy()
        # 左右腕
        kp_swapped[9], kp_swapped[10] = keypoints[10].copy(), keypoints[9].copy()
        # 左右髋
        kp_swapped[11], kp_swapped[12] = keypoints[12].copy(), keypoints[11].copy()
        # 左右膝
        kp_swapped[13], kp_swapped[14] = keypoints[14].copy(), keypoints[13].copy()
        # 左右踝
        kp_swapped[15], kp_swapped[16] = keypoints[16].copy(), keypoints[15].copy()
        
        return kp_swapped


class PropAwareSkeletonGenerator:
    """
    道具感知视觉骨架图生成器
    
    生成包含以下元素的骨架图:
    1. 18个标准人体关键点（彩色圆点）
    2. 骨架连接线（白色）
    3. 道具轮廓（青色，Cyan）
    """
    
    def __init__(self, config: Optional[SkeletonConfig] = None):
        self.config = config or SkeletonConfig()
        self.keypoint_processor = KeypointProcessor(config)
        
    def generate(
        self,
        keypoints: np.ndarray,
        prop_mask: Optional[np.ndarray] = None,
        image_size: Optional[Tuple[int, int]] = None,
        background: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        生成道具感知视觉骨架图
        
        Args:
            keypoints: 关键点数组 [N, 3] (x, y, confidence)
            prop_mask: 道具分割掩码 [H, W] 或 [H, W, 1]
            image_size: 输出图像尺寸 (width, height)
            background: 可选的背景图像
            
        Returns:
            骨架图 [H, W, 3] (BGR格式)
        """
        image_size = image_size or self.config.target_size
        width, height = image_size
        
        # 创建画布
        if background is not None:
            canvas = cv2.resize(background, (width, height))
            if canvas.ndim == 2:
                canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        else:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 1. 绘制道具轮廓（青色）
        if prop_mask is not None:
            canvas = self._draw_prop_contours(canvas, prop_mask)
        
        # 2. 绘制骨架连接线（白色）
        canvas = self._draw_skeleton(canvas, keypoints)
        
        # 3. 绘制关键点（彩色）
        canvas = self._draw_keypoints(canvas, keypoints)
        
        return canvas
    
    def _draw_prop_contours(
        self,
        canvas: np.ndarray,
        prop_mask: np.ndarray
    ) -> np.ndarray:
        """绘制道具轮廓"""
        # 确保掩码是二值的
        if prop_mask.ndim == 3:
            prop_mask = prop_mask[:, :, 0]
        
        # 调整掩码大小
        h, w = canvas.shape[:2]
        if prop_mask.shape != (h, w):
            prop_mask = cv2.resize(
                prop_mask.astype(np.uint8), 
                (w, h), 
                interpolation=cv2.INTER_NEAREST
            )
        
        # 二值化
        _, binary_mask = cv2.threshold(
            prop_mask.astype(np.uint8), 
            127, 255, 
            cv2.THRESH_BINARY
        )
        
        # 提取轮廓
        contours, _ = cv2.findContours(
            binary_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 绘制青色轮廓
        cv2.drawContours(
            canvas, 
            contours, 
            -1, 
            PROP_COLOR, 
            self.config.prop_thickness
        )
        
        return canvas
    
    def _draw_skeleton(
        self,
        canvas: np.ndarray,
        keypoints: np.ndarray
    ) -> np.ndarray:
        """绘制骨架连接线"""
        # 选择连接定义
        if self.config.use_neck and len(keypoints) > 17:
            connections = SKELETON_CONNECTIONS
        else:
            connections = SKELETON_CONNECTIONS_NO_NECK
        
        for (i, j) in connections:
            if i >= len(keypoints) or j >= len(keypoints):
                continue
                
            kp1, kp2 = keypoints[i], keypoints[j]
            
            # 检查置信度
            if kp1[2] < self.config.confidence_threshold:
                continue
            if kp2[2] < self.config.confidence_threshold:
                continue
            
            # 绘制连接线
            pt1 = (int(kp1[0]), int(kp1[1]))
            pt2 = (int(kp2[0]), int(kp2[1]))
            
            cv2.line(
                canvas, 
                pt1, 
                pt2, 
                SKELETON_LINE_COLOR, 
                self.config.skeleton_thickness
            )
        
        return canvas
    
    def _draw_keypoints(
        self,
        canvas: np.ndarray,
        keypoints: np.ndarray
    ) -> np.ndarray:
        """绘制关键点"""
        # 关键点到身体部位的映射
        keypoint_groups = {
            "head": [0, 1, 2, 3, 4],
            "torso": [5, 6, 11, 12, 17] if self.config.use_neck else [5, 6, 11, 12],
            "left_arm": [7, 9],
            "right_arm": [8, 10],
            "left_leg": [13, 15],
            "right_leg": [14, 16],
        }
        
        for group_name, indices in keypoint_groups.items():
            color = KEYPOINT_COLORS[group_name]
            
            for idx in indices:
                if idx >= len(keypoints):
                    continue
                    
                kp = keypoints[idx]
                if kp[2] < self.config.confidence_threshold:
                    continue
                
                center = (int(kp[0]), int(kp[1]))
                cv2.circle(
                    canvas, 
                    center, 
                    self.config.keypoint_radius, 
                    color, 
                    -1  # 填充圆
                )
        
        return canvas
    
    def generate_batch(
        self,
        keypoints_batch: List[np.ndarray],
        prop_masks: Optional[List[np.ndarray]] = None,
        image_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        批量生成骨架图
        
        Args:
            keypoints_batch: 关键点列表
            prop_masks: 道具掩码列表
            image_size: 输出尺寸
            
        Returns:
            批量骨架图 [B, H, W, 3]
        """
        batch = []
        for i, kp in enumerate(keypoints_batch):
            mask = prop_masks[i] if prop_masks else None
            skeleton_map = self.generate(kp, mask, image_size)
            batch.append(skeleton_map)
        
        return np.stack(batch, axis=0)


class SkeletonPreprocessor:
    """
    骨架图预处理器
    
    功能:
    1. 从图像/JSON加载骨架数据
    2. 标准化处理
    3. 转换为模型输入格式
    """
    
    def __init__(self, config: Optional[SkeletonConfig] = None):
        self.config = config or SkeletonConfig()
        self.keypoint_processor = KeypointProcessor(config)
        self.skeleton_generator = PropAwareSkeletonGenerator(config)
        
    def load_skeleton(
        self,
        source: Union[str, Path, np.ndarray, Dict]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        从不同来源加载骨架数据
        
        Args:
            source: 骨架数据来源
                - str/Path: JSON文件路径或图像路径
                - np.ndarray: 直接的关键点数组
                - Dict: 包含keypoints和可选prop_mask的字典
                
        Returns:
            (keypoints, prop_mask) 元组
        """
        prop_mask = None
        
        if isinstance(source, (str, Path)):
            source = Path(source)
            
            if source.suffix.lower() == '.json':
                # JSON格式
                with open(source, 'r') as f:
                    data = json.load(f)
                keypoints = data.get('keypoints', data)
                prop_mask = data.get('prop_mask', None)
                if prop_mask is not None:
                    prop_mask = np.array(prop_mask)
            else:
                # 图像格式（假设是预渲染的骨架图）
                skeleton_img = cv2.imread(str(source))
                keypoints = self._extract_keypoints_from_image(skeleton_img)
        
        elif isinstance(source, dict):
            keypoints = source.get('keypoints', source)
            prop_mask = source.get('prop_mask', None)
            if prop_mask is not None:
                prop_mask = np.array(prop_mask)
        
        else:
            keypoints = source
        
        # 解析关键点
        keypoints = self.keypoint_processor.parse_keypoints(keypoints)
        
        return keypoints, prop_mask
    
    def _extract_keypoints_from_image(
        self,
        skeleton_img: np.ndarray
    ) -> np.ndarray:
        """
        从骨架图像中提取关键点（简化实现）
        
        注意：实际应用中应使用姿态估计模型
        """
        # 这是一个简化的实现，实际使用时需要接入OpenPose等模型
        # 这里返回空关键点，表示需要外部提供关键点数据
        n_keypoints = 18 if self.config.use_neck else 17
        return np.zeros((n_keypoints, 3), dtype=np.float32)
    
    def preprocess(
        self,
        keypoints: np.ndarray,
        prop_mask: Optional[np.ndarray] = None,
        original_size: Optional[Tuple[int, int]] = None,
        augment: bool = False,
        seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        预处理骨架数据
        
        Args:
            keypoints: 关键点数组
            prop_mask: 道具掩码
            original_size: 原始图像尺寸 (width, height)
            augment: 是否进行数据增强
            seed: 随机种子
            
        Returns:
            预处理后的数据字典
        """
        original_size = original_size or self.config.target_size
        
        # 标准化关键点
        kp_normalized = self.keypoint_processor.normalize_keypoints(
            keypoints, 
            original_size, 
            self.config.target_size
        )
        
        # 数据增强
        aug_params = {}
        if augment:
            kp_normalized, aug_params = self.keypoint_processor.augment_keypoints(
                kp_normalized,
                self.config.target_size,
                seed
            )
            
            # 同步增强道具掩码
            if prop_mask is not None:
                prop_mask = self._augment_mask(prop_mask, aug_params)
        
        # 过滤低置信度关键点
        kp_filtered = self.keypoint_processor.filter_by_confidence(kp_normalized)
        
        # 生成骨架图
        skeleton_map = self.skeleton_generator.generate(
            kp_filtered,
            prop_mask,
            self.config.target_size
        )
        
        return {
            "keypoints": kp_filtered,
            "skeleton_map": skeleton_map,
            "prop_mask": prop_mask,
            "augmentation_params": aug_params
        }
    
    def _augment_mask(
        self,
        mask: np.ndarray,
        params: Dict
    ) -> np.ndarray:
        """同步增强掩码"""
        if not params:
            return mask
        
        h, w = mask.shape[:2]
        center = (w // 2, h // 2)
        
        # 构建变换矩阵
        scale = params.get("scale", 1.0)
        angle = params.get("rotation", 0.0)
        tx, ty = params.get("translation", (0, 0))
        flip = params.get("flip", False)
        
        # 旋转+缩放矩阵
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        
        # 应用变换
        mask_aug = cv2.warpAffine(
            mask.astype(np.uint8), 
            M, 
            (w, h),
            flags=cv2.INTER_NEAREST
        )
        
        # 翻转
        if flip:
            mask_aug = cv2.flip(mask_aug, 1)
        
        return mask_aug
    
    def to_tensor(
        self,
        skeleton_map: np.ndarray,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        转换骨架图为PyTorch张量
        
        Args:
            skeleton_map: 骨架图 [H, W, 3] (BGR)
            normalize: 是否归一化到[-1, 1]
            
        Returns:
            张量 [3, H, W]
        """
        # BGR -> RGB
        skeleton_rgb = cv2.cvtColor(skeleton_map, cv2.COLOR_BGR2RGB)
        
        # HWC -> CHW
        tensor = torch.from_numpy(skeleton_rgb).permute(2, 0, 1).float()
        
        # 归一化
        if normalize:
            tensor = tensor / 127.5 - 1.0  # [0, 255] -> [-1, 1]
        else:
            tensor = tensor / 255.0  # [0, 255] -> [0, 1]
        
        return tensor
    
    def batch_preprocess(
        self,
        sources: List[Union[str, Path, np.ndarray, Dict]],
        augment: bool = False,
        return_tensors: bool = True
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        批量预处理
        
        Args:
            sources: 数据源列表
            augment: 是否增强
            return_tensors: 是否返回张量
            
        Returns:
            批量处理结果
        """
        skeleton_maps = []
        keypoints_batch = []
        
        for i, source in enumerate(sources):
            kp, mask = self.load_skeleton(source)
            result = self.preprocess(
                kp, mask, 
                augment=augment, 
                seed=i if augment else None
            )
            
            skeleton_maps.append(result["skeleton_map"])
            keypoints_batch.append(result["keypoints"])
        
        skeleton_maps = np.stack(skeleton_maps, axis=0)
        keypoints_batch = np.stack(keypoints_batch, axis=0)
        
        if return_tensors:
            # 批量转换为张量
            tensors = []
            for sm in skeleton_maps:
                tensors.append(self.to_tensor(sm))
            skeleton_tensors = torch.stack(tensors, dim=0)
            
            return {
                "skeleton_maps": skeleton_tensors,
                "keypoints": torch.from_numpy(keypoints_batch)
            }
        
        return {
            "skeleton_maps": skeleton_maps,
            "keypoints": keypoints_batch
        }


class PropDetector:
    """
    道具检测器（占位实现）
    
    实际应用中应接入分割模型（如SAM、YOLO-Seg等）
    """
    
    def __init__(self, config: Optional[SkeletonConfig] = None):
        self.config = config or SkeletonConfig()
        
    def detect(
        self,
        image: np.ndarray,
        keypoints: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        检测图像中的道具区域
        
        Args:
            image: 输入图像 [H, W, 3]
            keypoints: 可选的关键点信息（用于排除人体区域）
            
        Returns:
            道具掩码 [H, W]
        """
        # 占位实现：返回空掩码
        # 实际应用中需要：
        # 1. 使用目标检测/分割模型检测道具
        # 2. 基于规则排除人体区域
        # 3. 后处理优化掩码
        
        h, w = image.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)


# ==================== 工具函数 ====================

def create_preprocessor(
    target_size: Tuple[int, int] = (1024, 1024),
    enable_augmentation: bool = True,
    **kwargs
) -> SkeletonPreprocessor:
    """
    创建骨架预处理器的工厂函数
    
    Args:
        target_size: 目标图像尺寸
        enable_augmentation: 是否启用数据增强
        **kwargs: 其他配置参数
        
    Returns:
        SkeletonPreprocessor实例
    """
    config = SkeletonConfig(
        target_size=target_size,
        enable_augmentation=enable_augmentation,
        **kwargs
    )
    return SkeletonPreprocessor(config)


def create_skeleton_generator(
    target_size: Tuple[int, int] = (1024, 1024),
    **kwargs
) -> PropAwareSkeletonGenerator:
    """
    创建骨架图生成器的工厂函数
    """
    config = SkeletonConfig(target_size=target_size, **kwargs)
    return PropAwareSkeletonGenerator(config)


def visualize_skeleton(
    keypoints: np.ndarray,
    prop_mask: Optional[np.ndarray] = None,
    image_size: Tuple[int, int] = (512, 512),
    background: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    可视化骨架（便捷函数）
    
    Args:
        keypoints: 关键点数组 [N, 3]
        prop_mask: 道具掩码
        image_size: 输出尺寸
        background: 背景图像
        
    Returns:
        可视化图像 (RGB格式)
    """
    generator = PropAwareSkeletonGenerator()
    skeleton_bgr = generator.generate(keypoints, prop_mask, image_size, background)
    # BGR -> RGB
    return cv2.cvtColor(skeleton_bgr, cv2.COLOR_BGR2RGB)


def save_skeleton_image(
    skeleton_map: np.ndarray,
    save_path: Union[str, Path],
    is_bgr: bool = True
) -> None:
    """保存骨架图像"""
    if is_bgr:
        cv2.imwrite(str(save_path), skeleton_map)
    else:
        cv2.imwrite(str(save_path), cv2.cvtColor(skeleton_map, cv2.COLOR_RGB2BGR))


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("测试骨架预处理模块...")
    
    # 创建测试关键点（模拟T-pose）
    test_keypoints = np.array([
        # x, y, confidence
        [256, 100, 1.0],   # 0: nose
        [240, 90, 1.0],    # 1: left_eye
        [272, 90, 1.0],    # 2: right_eye
        [220, 100, 1.0],   # 3: left_ear
        [292, 100, 1.0],   # 4: right_ear
        [200, 180, 1.0],   # 5: left_shoulder
        [312, 180, 1.0],   # 6: right_shoulder
        [140, 180, 1.0],   # 7: left_elbow
        [372, 180, 1.0],   # 8: right_elbow
        [80, 180, 1.0],    # 9: left_wrist
        [432, 180, 1.0],   # 10: right_wrist
        [220, 320, 1.0],   # 11: left_hip
        [292, 320, 1.0],   # 12: right_hip
        [220, 420, 1.0],   # 13: left_knee
        [292, 420, 1.0],   # 14: right_knee
        [220, 500, 1.0],   # 15: left_ankle
        [292, 500, 1.0],   # 16: right_ankle
    ], dtype=np.float32)
    
    # 创建测试道具掩码（模拟剑）
    test_prop_mask = np.zeros((512, 512), dtype=np.uint8)
    # 在右手位置绘制剑形道具
    cv2.rectangle(test_prop_mask, (420, 150), (500, 220), 255, -1)
    
    print("1. 测试KeypointProcessor...")
    kp_processor = KeypointProcessor()
    kp_parsed = kp_processor.parse_keypoints(test_keypoints)
    print(f"   解析后关键点形状: {kp_parsed.shape}")
    
    kp_normalized = kp_processor.normalize_keypoints(kp_parsed, (512, 512), (1024, 1024))
    print(f"   标准化后关键点形状: {kp_normalized.shape}")
    
    kp_augmented, aug_params = kp_processor.augment_keypoints(kp_parsed, (512, 512), seed=42)
    print(f"   增强参数: {aug_params}")
    
    print("\n2. 测试PropAwareSkeletonGenerator...")
    generator = PropAwareSkeletonGenerator()
    skeleton_map = generator.generate(kp_parsed, test_prop_mask, (512, 512))
    print(f"   生成骨架图形状: {skeleton_map.shape}")
    print(f"   骨架图范围: [{skeleton_map.min()}, {skeleton_map.max()}]")
    
    print("\n3. 测试SkeletonPreprocessor...")
    preprocessor = SkeletonPreprocessor()
    result = preprocessor.preprocess(
        test_keypoints, 
        test_prop_mask, 
        original_size=(512, 512),
        augment=True,
        seed=42
    )
    print(f"   预处理结果键: {list(result.keys())}")
    print(f"   骨架图形状: {result['skeleton_map'].shape}")
    
    print("\n4. 测试张量转换...")
    tensor = preprocessor.to_tensor(result['skeleton_map'])
    print(f"   张量形状: {tensor.shape}")
    print(f"   张量范围: [{tensor.min():.2f}, {tensor.max():.2f}]")
    
    print("\n5. 测试批量处理...")
    batch_result = preprocessor.batch_preprocess(
        [{"keypoints": test_keypoints, "prop_mask": test_prop_mask}] * 4,
        augment=True,
        return_tensors=True
    )
    print(f"   批量骨架图张量形状: {batch_result['skeleton_maps'].shape}")
    print(f"   批量关键点张量形状: {batch_result['keypoints'].shape}")
    
    print("\n✓ 所有测试通过!")