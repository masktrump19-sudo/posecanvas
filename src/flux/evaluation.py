"""
Flux Model Evaluation Metrics

本模块实现以下四个评估指标 (Evaluation Metrics)：
1. OKS (Object Keypoint Similarity): 姿势吻合度
2. FID (Fréchet Inception Distance): 图像分布距离 (质量)
3. CLIP Score: 语义一致性
4. LBR Variance (Limb Bone Ratio Variance): 人体解剖学稳定性

Dependencies:
    - torch
    - torchvision
    - numpy
    - transformers (for CLIP)
    - torchmetrics (for FID)
    - scipy (optional)
    - opencv-python
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
from PIL import Image

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.multimodal.clip_score import CLIPScore
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    print("Warning: torchmetrics or transformers not installed. Some metrics may not work.")

# COCO Keypoint Constants for OKS
# SIGMAS: 关键点尺度的标准差 (Standard deviation)
# 对应: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
COCO_SIGMAS = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
    0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
]) / 10.0

COCO_VARS = (COCO_SIGMAS * 2)**2

class Evaluator:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._init_models()
        
    def _init_models(self):
        """Initialize evaluation models lazy or on startup"""
        # FID
        try:
            self.fid = FrechetInceptionDistance(feature=2048).to(self.device)
        except Exception as e:
            print(f"FID init failed: {e}")
            self.fid = None

        # CLIP
        try:
            # 使用 torchmetrics 或 transformers 直接实现
            self.clip_model_name = "openai/clip-vit-base-patch32"
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        except Exception as e:
            print(f"CLIP init failed: {e}")
            self.clip_model = None

    # ==================== 1. OKS (Object Keypoint Similarity) ====================
    
    def compute_oks(self, gt_keypoints: np.ndarray, pred_image: Union[str, np.ndarray]) -> float:
        """
        计算 OKS 指标
        
        Args:
            gt_keypoints: 输入骨架的 Ground Truth 关键点 [N, 17, 3] (x, y, v) 或 [17, 3]
            pred_image: 生成的图像 (Path or Numpy array)
            
        Returns:
            oks_score: OKS 分数 [0, 1]
        """
        # 1. 从生成图像中检测关键点
        # 注意: 这里需要集成实际的 Pose Estimator (如 DWPose)
        # 暂时使用 mock 函数或 API 接口
        pred_keypoints = self._detect_keypoints(pred_image)
        
        if gt_keypoints.ndim == 2:
            gt_keypoints = gt_keypoints[np.newaxis, ...] # [1, 17, 3]
            
        if pred_keypoints is None:
            return 0.0
            
        # 2. 计算 OKS
        # OKS = sum(exp(-di^2 / 2s^2k^2) * delta(vi>0)) / sum(delta(vi>0))
        
        # 假设 batch size = 1 进行计算，或取最大匹配
        gt = gt_keypoints[0] # [17, 3]
        pred = pred_keypoints[0] # [17, 3] (take first person detected)
        
        # 提取坐标和可见性
        xg, yg, vg = gt[:, 0], gt[:, 1], gt[:, 2]
        xp, yp, vp = pred[:, 0], pred[:, 1], pred[:, 2]
        
        # 计算对象尺度 (Scale s) - 简单的就是边界框面积的平方根
        # 这里简化计算：使用可见关键点的覆盖面积
        if np.sum(vg > 0) > 0:
            xmin, xmax = np.min(xg[vg>0]), np.max(xg[vg>0])
            ymin, ymax = np.min(yg[vg>0]), np.max(yg[vg>0])
            area = (xmax - xmin) * (ymax - ymin)
            s = np.sqrt(area) + 1e-6
        else:
            s = 1.0

        # 计算距离
        dx = xg - xp
        dy = yg - yp
        d2 = dx**2 + dy**2
        
        # 计算指数项
        # 确保关键点数量匹配 COCO_VARS (17个)
        num_kpts = min(len(d2), len(COCO_VARS))
        e = d2[:num_kpts] / (2 * s**2 * COCO_VARS[:num_kpts] + 1e-9)
        
        # 仅考虑 GT 可见的关键点
        visible = (vg[:num_kpts] > 0)
        
        if np.sum(visible) == 0:
            return 0.0
            
        oks = np.sum(np.exp(-e)[visible]) / np.sum(visible)
        return float(oks)

    def _detect_keypoints(self, image_input) -> np.ndarray:
        """
        使用 Pose Estimator 检测关键点
        此处应集成 DWPose / OpenPose / MMPose
        """
        # TODO: Implement DWPose inference
        # This is a placeholder. In a real scenario, load DWPoseOnnx or similar.
        # return np.zeros((1, 17, 3)) 
        
        # 如果无法检测，返回 None 或随机 (供测试)
        print("Warning: Pose detector not implemented. Returning dummy keypoints.")
        return np.random.rand(1, 17, 3) * 512 

    # ==================== 2. FID (Fréchet Inception Distance) ====================

    def update_fid(self, real_imgs: torch.Tensor, fake_imgs: torch.Tensor):
        """
        更新 FID 统计信息
        Args:
            real_imgs: [B, C, H, W], normalized [0, 1] or [0, 255] uint8
            fake_imgs: [B, C, H, W]
        """
        if self.fid is None: return
        
        # torchmetrics FID expects uint8 [0, 255]
        if real_imgs.dtype == torch.float32:
            real_imgs = (real_imgs * 255).byte()
        if fake_imgs.dtype == torch.float32:
            fake_imgs = (fake_imgs * 255).byte()
            
        self.fid.update(real_imgs, real=True)
        self.fid.update(fake_imgs, real=False)
        
    def compute_fid(self) -> float:
        if self.fid is None: return 0.0
        return float(self.fid.compute())
        
    def reset_fid(self):
        if self.fid is not None:
            self.fid.reset()

    # ==================== 3. CLIP Score ====================
    
    def compute_clip_score(self, image: Union[Image.Image, np.ndarray], text: str) -> float:
        """
        计算 CLIP Score (Semantic Consistency)
        """
        if self.clip_model is None: return 0.0
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        inputs = self.clip_processor(
            text=[text], 
            images=image, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            # image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            # text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            # cosine_sim = torch.matmul(image_embeds, text_embeds.t())
            
            # 使用 logits_per_image 自带的 scale
            logits_per_image = outputs.logits_per_image # this is cosine * 100
            score = logits_per_image.item()
            
        return max(score, 0.0)

    # ==================== 4. LBR Variance (Limb Bone Ratio Variance) ====================
    
    def compute_lbr_variance(self, keypoints: np.ndarray) -> float:
        """
        计算肢体骨骼比例方差 (LBR Variance)
        衡量解剖学稳定性。
        
        Logic:
        1. 计算四肢的上肢(Upper)和下肢(Lower)长度
        2. 计算比率 r = Lower / Upper
        3. 计算这些比率的方差 (衡量左右对称性和整体协调性) 或 与标准比率(如1.0)的偏差
        
        COCO Indices:
        - Arm Left:  Shoulder(5)-Elbow(7)-Wrist(9)
        - Arm Right: Shoulder(6)-Elbow(8)-Wrist(10)
        - Leg Left:  Hip(11)-Knee(13)-Ankle(15)
        - Leg Right: Hip(12)-Knee(14)-Ankle(16)
        """
        kpts = keypoints # [17, 3] or [N, 17, 3]
        if kpts.ndim == 3: kpts = kpts[0]
        
        # Helper to calc dist
        def dist(idx1, idx2):
            p1 = kpts[idx1, :2]
            p2 = kpts[idx2, :2]
            # 检查置信度
            if kpts.shape[-1] > 2:
                if kpts[idx1, 2] == 0 or kpts[idx2, 2] == 0:
                    return None
            return np.linalg.norm(p1 - p2)

        ratios = []
        
        # Define limbs: (Upper_Start, Upper_End/Lower_Start, Lower_End)
        limbs = [
            (5, 7, 9),   # Left Arm
            (6, 8, 10),  # Right Arm
            (11, 13, 15), # Left Leg
            (12, 14, 16)  # Right Leg
        ]
        
        for (idx1, idx2, idx3) in limbs:
            len_upper = dist(idx1, idx2)
            len_lower = dist(idx2, idx3)
            
            if len_upper is not None and len_lower is not None and len_upper > 0:
                r = len_lower / len_upper
                ratios.append(r)
                
        if not ratios:
            return 0.0 # No valid limbs
            
        # 计算方差
        # 如果追求"稳定性"，我们希望比率一致（例如所有的比率都接近特定值，或者左右对称）
        # 这里直接计算比率的方差
        variance = np.var(ratios)
        
        return float(variance)

# Example Usage
if __name__ == "__main__":
    evaluator = Evaluator(device="cpu") # use cpu for test
    
    # Mock data
    dummy_gt_kpts = np.random.rand(17, 3) * 100
    dummy_gt_kpts[:, 2] = 1 # all visible
    dummy_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    dummy_txt = "A girl dancing"
    
    print("\n--- Testing Metrics ---")
    
    # 1. OKS
    # Note: _detect_keypoints returns random, so score will be low/random
    oks = evaluator.compute_oks(dummy_gt_kpts, dummy_img)
    print(f"OKS: {oks:.4f}")
    
    # 2. FID
    # Need batch of images
    real_batch = torch.randint(0, 255, (2, 3, 299, 299), dtype=torch.uint8)
    fake_batch = torch.randint(0, 255, (2, 3, 299, 299), dtype=torch.uint8)
    evaluator.update_fid(real_batch, fake_batch)
    # fid = evaluator.compute_fid() # Requires inception weights, might be slow/fail without net
    # print(f"FID: {fid:.4f}")
    
    # 3. CLIP Score
    clip_score = evaluator.compute_clip_score(dummy_img, dummy_txt)
    print(f"CLIP Score: {clip_score:.4f}")
    
    # 4. LBR Variance
    # Construct a skeleton with perfect ratios (1.0)
    # Shoulder(0,0), Elbow(0,10), Wrist(0,20) -> Ratio 1.0
    perfect_kpts = np.zeros((17, 3))
    perfect_kpts[:, 2] = 1
    # Left Arm
    perfect_kpts[5] = [0, 0, 1]
    perfect_kpts[7] = [0, 10, 1]
    perfect_kpts[9] = [0, 20, 1] 
    # Right Arm
    perfect_kpts[6] = [10, 0, 1]
    perfect_kpts[8] = [10, 10, 1]
    perfect_kpts[10] = [10, 20, 1]
    
    lbr = evaluator.compute_lbr_variance(perfect_kpts)
    print(f"LBR Variance (Perfect): {lbr:.6f}") # Should be 0.0
