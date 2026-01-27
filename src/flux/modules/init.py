"""
数据加载与预处理模块

包含数据集定义、数据加载和骨架预处理功能
"""

from .dataset import (
    AnimePoseDataset,
    LatentAnimePoseDataset,
    create_dataloader,
)

from .preprocess import (
    SkeletonConfig,
    KeypointProcessor,
    PropAwareSkeletonGenerator,
    SkeletonPreprocessor,
    PropDetector,
    create_preprocessor,
    create_skeleton_generator,
    visualize_skeleton,
    save_skeleton_image,
    COCO_KEYPOINTS,
    SKELETON_CONNECTIONS,
)

__all__ = [
    # Dataset
    "AnimePoseDataset",
    "LatentAnimePoseDataset",
    "create_dataloader",
    # Preprocess
    "SkeletonConfig",
    "KeypointProcessor",
    "PropAwareSkeletonGenerator",
    "SkeletonPreprocessor",
    "PropDetector",
    "create_preprocessor",
    "create_skeleton_generator",
    "visualize_skeleton",
    "save_skeleton_image",
    "COCO_KEYPOINTS",
    "SKELETON_CONNECTIONS",
]