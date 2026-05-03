"""
航拍红外图像专用数据增强

设计原则：
1. 红外图像是单通道灰度（辐亮度/温度映射），不能做RGB颜色增强
2. 航拍小目标（通常<32x32像素）需要保持位置精度
3. 所有几何变换必须保证标签精确对齐
4. 红外物理特性：亮度分布反映热辐射，需模拟环境变化
"""
import numpy as np
import random
import math
from typing import Tuple, Optional, List
import cv2


class BaseAugmentation:
    """基础数据增强基类"""

    @staticmethod
    def _to_hwc(image: np.ndarray) -> np.ndarray:
        """[C, H, W] → [H, W, C]"""
        return np.transpose(image, (1, 2, 0))

    @staticmethod
    def _to_chw(image: np.ndarray) -> np.ndarray:
        """[H, W] or [H, W, C] → [C, H, W]"""
        if image.ndim == 2:
            return image[np.newaxis, :, :]
        return np.transpose(image, (2, 0, 1))

    def apply(self, image: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            image: [C, H, W] numpy array (uint8)
            labels: [N, 5] (class_id, x_center, y_center, width, height) 归一化坐标
        Returns:
            augmented image and labels
        """
        return image, labels


class RandomHorizontalFlip(BaseAugmentation):
    """水平翻转 - 对航拍图像物理对称性合理"""
    def __init__(self, p: float = 0.5):
        self.p = p

    def apply(self, image: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            # 水平翻转：axis=2 (width维度)
            image = np.flip(image, axis=2).copy()
            if len(labels) > 0:
                labels[:, 1] = 1.0 - labels[:, 1]  # x_center -> 1 - x_center
        return image, labels


class RandomVerticalFlip(BaseAugmentation):
    """垂直翻转 - 对航拍图像（俯视）物理对称性合理"""
    def __init__(self, p: float = 0.5):
        self.p = p

    def apply(self, image: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            image = np.flip(image, axis=1).copy()
            if len(labels) > 0:
                labels[:, 2] = 1.0 - labels[:, 2]  # y_center -> 1 - y_center
        return image, labels


class RandomRotate90(BaseAugmentation):
    """
    90°整数倍旋转 - 适合航拍图像
    ✅ 对红外小目标无损（旋转后像素精确对齐）
    ✅ 标签可精确计算
    """
    def __init__(self, p: float = 0.5):
        self.p = p

    def apply(self, image: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            k = random.choice([1, 2, 3])  # 90°, 180°, 270°
            # 旋转图像
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            if len(labels) > 0:
                h, w = 1.0, 1.0  # 归一化坐标
                x, y, w_box, h_box = labels[:, 1], labels[:, 2], labels[:, 3], labels[:, 4]
                if k == 1:  # 90° 顺时针
                    labels[:, 1] = y
                    labels[:, 2] = 1.0 - x - w_box
                    labels[:, 3] = h_box
                    labels[:, 4] = w_box
                elif k == 2:  # 180°
                    labels[:, 1] = 1.0 - x - w_box
                    labels[:, 2] = 1.0 - y - h_box
                elif k == 3:  # 270° 顺时针 (90°逆时针)
                    labels[:, 1] = 1.0 - y - h_box
                    labels[:, 2] = x
                    labels[:, 3] = h_box
                    labels[:, 4] = w_box
        return image, labels


class RandomBrightnessContrast(BaseAugmentation):
    """
    亮度/对比度抖动 - 红外图像最关键增强

    原理：红外图像反映目标热辐射。环境温度变化、昼夜温差、太阳照射角度
    都会导致目标与背景的温差变化，表现为图像亮度和对比度的变化。
    
    红外特有的考虑：
    - 红外图像是16bit raw转8bit，不同场景的动态范围差异大
    - 目标温度高于背景时表现为亮斑（白热模式）
    - 对比度变化模拟不同天气条件下的热传导差异
    """
    def __init__(self, brightness_delta: float = 0.3, contrast_delta: float = 0.3, p: float = 0.7):
        """
        Args:
            brightness_delta: 亮度偏移范围 [1-b, 1+b]
            contrast_delta: 对比度缩放范围 [1-c, 1+c]
            p: 应用概率
        """
        self.brightness_delta = brightness_delta
        self.contrast_delta = contrast_delta
        self.p = p

    def apply(self, image: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            alpha = 1.0 + random.uniform(-self.contrast_delta, self.contrast_delta)
            beta = random.uniform(-self.brightness_delta, self.brightness_delta) * 255.0
            
            # 对每个通道应用相同变换（红外3通道复制值应一致）
            img_float = image.astype(np.float32)
            img_float = img_float * alpha + beta
            image = np.clip(img_float, 0, 255).astype(np.uint8)
        return image, labels


class RandomNoise(BaseAugmentation):
    """
    红外传感器噪声模拟

    红外探测器的主要噪声类型：
    1. 高斯噪声 - 焦平面阵列读出噪声
    2. 散粒噪声（泊松） - 光子计数统计涨落
    3. 固定模式噪声 - 由非均匀校正残留（NUC已部分补偿）
    
    对小目标影响：噪声可能淹没弱目标，增强后模型学会区分噪声和目标。
    """
    def __init__(self, noise_type: str = 'gaussian', gaussian_std: float = 0.05, 
                 p: float = 0.5):
        """
        Args:
            noise_type: 'gaussian' | 'poisson' | 'salt_pepper'
            gaussian_std: 高斯噪声标准差 (相对于255)
            p: 应用概率
        """
        self.noise_type = noise_type
        self.gaussian_std = gaussian_std
        self.p = p

    def apply(self, image: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            img_float = image.astype(np.float32)
            
            if self.noise_type == 'gaussian':
                std = random.uniform(0, self.gaussian_std) * 255.0
                noise = np.random.randn(*image.shape).astype(np.float32) * std
                img_float = img_float + noise
                
            elif self.noise_type == 'poisson':
                # 泊松噪声：噪声强度与信号强度成正比
                # 红外图像中信号越强（温度越高），光子噪声越大
                scale = random.uniform(10, 30)
                noisy = np.random.poisson(img_float / scale) * scale
                img_float = noisy.astype(np.float32)
                
            elif self.noise_type == 'salt_pepper':
                salt_prob = random.uniform(0.001, 0.01)
                pepper_prob = random.uniform(0.001, 0.01)
                mask_salt = np.random.random(image.shape) < salt_prob
                mask_pepper = np.random.random(image.shape) < pepper_prob
                img_float[mask_salt] = 255.0
                img_float[mask_pepper] = 0.0
                
            image = np.clip(img_float, 0, 255).astype(np.uint8)
        return image, labels


class RandomBlur(BaseAugmentation):
    """
    模糊模拟 - 航拍红外特有的光学效应

    原因：
    1. 大气湍流导致的热晕效应（红外波段更明显）
    2. 无人机/飞机平台振动
    3. 光学系统离焦
    4. 运动模糊（目标或平台移动）
    
    对训练的好处：模型学习从模糊图像中仍能检测到目标，提高鲁棒性。
    """
    def __init__(self, max_kernel: int = 5, blur_type: str = 'gaussian', p: float = 0.3):
        self.max_kernel = max_kernel
        self.blur_type = blur_type
        self.p = p

    def apply(self, image: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            k = random.choice([3, 5]) if self.max_kernel >= 5 else 3
            C = image.shape[0]

            if self.blur_type == 'gaussian':
                sigma = random.uniform(0.5, 2.0)
                for c in range(C):
                    image[c] = cv2.GaussianBlur(image[c], (k, k), sigma)
            elif self.blur_type == 'average':
                for c in range(C):
                    image[c] = cv2.blur(image[c], (k, k))
            elif self.blur_type == 'motion':
                size = random.choice([3, 5, 7])
                angle = random.uniform(0, 180)
                kernel_motion = np.zeros((size, size))
                kernel_motion[int((size-1)/2), :] = 1
                kernel_motion = cv2.warpAffine(
                    kernel_motion,
                    cv2.getRotationMatrix2D((size/2-0.5, size/2-0.5), angle, 1.0),
                    (size, size)
                )
                kernel_motion /= kernel_motion.sum()
                for c in range(C):
                    image[c] = cv2.filter2D(image[c], -1, kernel_motion)
        return image, labels


class RandomScale(BaseAugmentation):
    """
    随机缩放 - 模拟不同航高
    
    航拍中目标尺寸与飞行高度成正比。多尺度训练让模型适应不同尺度目标。
    
    注意：红外小目标（如3x3像素）缩小后可能消失，需用插值保护。
    """
    def __init__(self, scale_range: Tuple[float, float] = (0.5, 1.5), p: float = 0.5):
        self.scale_range = scale_range
        self.p = p

    def apply(self, image: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            h_img, w_img = image.shape[1], image.shape[2]
            scale = random.uniform(*self.scale_range)
            new_w, new_h = int(w_img * scale), int(h_img * scale)

            new_w = max(new_w, 32)
            new_h = max(new_h, 32)

            # Scale each channel independently (avoids CHW↔HWC transpose)
            C = image.shape[0]
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            resized = np.empty((C, new_h, new_w), dtype=np.uint8)
            for c in range(C):
                resized[c] = cv2.resize(image[c], (new_w, new_h), interpolation=interp)

            image = resized
            # 缩放到目标尺寸（letterbox）
            image, labels = self._letterbox(image, labels, (h_img, w_img))
        return image, labels

    def _letterbox(self, image: np.ndarray, labels: np.ndarray,
                   target_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """保持宽高比的缩放填充（逐通道处理，无 transpose）"""
        C, h, w = image.shape
        target_h, target_w = target_size

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize each channel independently
        resized = np.empty((C, new_h, new_w), dtype=np.uint8)
        for c in range(C):
            resized[c] = cv2.resize(image[c], (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        dw = (target_w - new_w) / 2
        dh = (target_h - new_h) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        # Pad with zeros (slice assignment, no copyMakeBorder needed)
        image = np.zeros((C, target_h, target_w), dtype=np.uint8)
        image[:, top:top + new_h, left:left + new_w] = resized

        if len(labels) > 0:
            labels[:, 1] = (labels[:, 1] * w * scale + dw) / target_w
            labels[:, 2] = (labels[:, 2] * h * scale + dh) / target_h
            labels[:, 3] = labels[:, 3] * scale * w / target_w
            labels[:, 4] = labels[:, 4] * scale * h / target_h

            labels[:, 1:] = np.clip(labels[:, 1:], 0.0, 1.0)
            valid = (labels[:, 3] > 0.001) & (labels[:, 4] > 0.001)
            labels = labels[valid]

        return image, labels


class RandomTranslate(BaseAugmentation):
    """
    随机平移 - 模拟航拍目标位置偏移

    航拍时目标可能出现在图像的任何位置，平移增强让模型学习位置不变性。
    对红外小目标尤其重要：模型不应该依赖目标在图像中的绝对位置。
    """
    def __init__(self, max_translate: float = 0.1, p: float = 0.5):
        """
        Args:
            max_translate: 最大平移比例 (相对于图像尺寸)
        """
        self.max_translate = max_translate
        self.p = p

    def apply(self, image: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            h, w = image.shape[1], image.shape[2]
            tx = random.uniform(-self.max_translate, self.max_translate) * w
            ty = random.uniform(-self.max_translate, self.max_translate) * h
            
            M = np.float32([[1, 0, tx], [0, 1, ty]])

            # 平移图像（逐通道处理，避免 CHW→HWC→CHW 转换）
            for c in range(image.shape[0]):
                image[c] = cv2.warpAffine(image[c], M, (w, h),
                                          borderMode=cv2.BORDER_REFLECT)
            
            # 平移标签
            if len(labels) > 0:
                labels[:, 1] += tx / w
                labels[:, 2] += ty / h
                
                # 裁剪并过滤（注意 _filter_labels 返回新数组）
                labels = self._filter_labels(labels)
        return image, labels

    def _filter_labels(self, labels: np.ndarray) -> np.ndarray:
        """裁剪偏移出边界的框"""
        if len(labels) == 0:
            return labels
        
        # 计算绝对坐标
        x1 = labels[:, 1] - labels[:, 3] / 2
        y1 = labels[:, 2] - labels[:, 4] / 2
        x2 = labels[:, 1] + labels[:, 3] / 2
        y2 = labels[:, 2] + labels[:, 4] / 2
        
        # 裁剪
        x1 = np.clip(x1, 0, 1)
        y1 = np.clip(y1, 0, 1)
        x2 = np.clip(x2, 0, 1)
        y2 = np.clip(y2, 0, 1)
        
        labels[:, 1] = (x1 + x2) / 2
        labels[:, 2] = (y1 + y2) / 2
        labels[:, 3] = x2 - x1
        labels[:, 4] = y2 - y1
        
        # 保留有效框
        valid = (labels[:, 3] > 0.005) & (labels[:, 4] > 0.005)
        return labels[valid]


class Mosaic(BaseAugmentation):
    """
    Mosaic增强 - 航拍红外小目标的王牌增强

    为什么对红外小目标特别有效：
    1. 将4张图拼成1张，相当于batch内的小目标数量增加4倍
    2. 小目标被放在不同背景上下文中，学习丰富的特征
    3. 强制模型在图像的不同区域检测，减少位置偏差
    4. 自然引入尺度变化
    5. 对小目标密集场景，增加了目标间的交互学习
    """
    def __init__(self, p: float = 0.5, target_size: int = 640,
                 dataset: Optional[object] = None):
        """
        Args:
            p: 应用概率
            target_size: 输出图像尺寸
            dataset: 数据集引用，用于获取随机样本
        """
        self.p = p
        self.target_size = target_size
        self.dataset = dataset

    def apply(self, image: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Mosaic增强：将当前图 + 3张随机图拼成1张"""
        if random.random() >= self.p or self.dataset is None:
            return image, labels

        C, H, W = image.shape
        target_h = target_w = self.target_size

        # 随机选中心分裂点
        cx = random.randint(W // 4, 3 * W // 4)
        cy = random.randint(H // 4, 3 * H // 4)

        # 收集4张图：当前图 + 3张随机图
        ds_len = len(self.dataset)
        images = [image]
        all_labels = [labels]

        indices = [random.randint(0, ds_len - 1) for _ in range(3)]
        from .dataset import YOLODataset
        for idx in indices:
            YOLODataset._recursive_guard = True
            try:
                img_t, lbl_t, _ = self.dataset[idx]
            finally:
                YOLODataset._recursive_guard = False
            other_img = img_t.numpy() if hasattr(img_t, 'numpy') else np.array(img_t)
            other_lbl = lbl_t.numpy() if hasattr(lbl_t, 'numpy') and len(lbl_t) > 0 else np.zeros((0, 5), dtype=np.float32)
            images.append(other_img)
            all_labels.append(other_lbl)

        # 拼成4象限
        # 图像顺序: 左上=0, 右上=1, 左下=2, 右下=3
        out_img = np.zeros((C, target_h, target_w), dtype=np.uint8)
        out_labels = []

        quad_info = [
            # (img_idx, src_crop, dst_region, h_scale, w_scale, h_off, w_off)
            # 左上: 图片0 取 [0:cy, 0:cx] -> [0:cy_out, 0:cx_out]
            (0, (0, cy, 0, cx), (0, 0)),
            # 右上: 图片1 取 [0:cy, cx:W] -> [0:cy_out, cx_out:target_w]
            (1, (0, cy, cx, W), (0, cx)),
            # 左下: 图片2 取 [cy:H, 0:cx] -> [cy_out:target_h, 0:cx_out]
            (2, (cy, H, 0, cx), (cy, 0)),
            # 右下: 图片3 取 [cy:H, cx:W] -> [cy_out:target_h, cx_out:target_w]
            (3, (cy, H, cx, W), (cy, cx)),
        ]

        def _resize_crop(src_img, y1, y2, x1, x2, out_h, out_w):
            """裁剪并缩放到目标大小"""
            crop = src_img[:, y1:y2, x1:x2]  # [C, h, w]
            if crop.shape[1] == 0 or crop.shape[2] == 0:
                return None
            return np.stack([cv2.resize(crop[c], (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                             for c in range(C)], axis=0)

        for img_idx, (y1, y2, x1, x2), (dst_y, dst_x) in quad_info:
            src = images[img_idx]
            src_h, src_w = src.shape[1], src.shape[2]
            # 裁剪区域
            c_y1 = max(0, y1)
            c_y2 = min(src_h, y2)
            c_x1 = max(0, x1)
            c_x2 = min(src_w, x2)

            crop_h = c_y2 - c_y1
            crop_w = c_x2 - c_x1
            if crop_h <= 0 or crop_w <= 0:
                continue

            # 目标区域大小
            dst_h = target_h // 2
            dst_w = target_w // 2
            if img_idx > 1:
                dst_h = target_h - cy
            if img_idx % 2 == 1:
                dst_w = target_w - cx

            resized = _resize_crop(src, c_y1, c_y2, c_x1, c_x2, dst_h, dst_w)
            if resized is not None:
                out_img[:, dst_y:dst_y+dst_h, dst_x:dst_x+dst_w] = resized

            # 转换标签
            lbl = all_labels[img_idx]
            if len(lbl) > 0:
                for row in lbl:
                    cls_id = int(row[0])
                    xc, yc, bw, bh = row[1:5]
                    # 原图归一化坐标 -> 绝对坐标
                    abs_x = xc * src_w
                    abs_y = yc * src_h
                    abs_w = bw * src_w
                    abs_h = bh * src_h

                    # 检查是否在裁剪区域内
                    x1_a = abs_x - abs_w / 2
                    y1_a = abs_y - abs_h / 2
                    x2_a = abs_x + abs_w / 2
                    y2_a = abs_y + abs_h / 2

                    # 裁剪到裁剪区域
                    x1_c = max(x1_a, c_x1)
                    y1_c = max(y1_a, c_y1)
                    x2_c = min(x2_a, c_x2)
                    y2_c = min(y2_a, c_y2)

                    if x2_c - x1_c <= 1 or y2_c - y1_c <= 1:
                        continue  # 裁剪后太小，丢弃

                    # 映射到目标区域
                    new_x1 = (x1_c - c_x1) / crop_w * dst_w + dst_x
                    new_y1 = (y1_c - c_y1) / crop_h * dst_h + dst_y
                    new_x2 = (x2_c - c_x1) / crop_w * dst_w + dst_x
                    new_y2 = (y2_c - c_y1) / crop_h * dst_h + dst_y

                    # 归一化到输出图像
                    out_xc = (new_x1 + new_x2) / 2 / target_w
                    out_yc = (new_y1 + new_y2) / 2 / target_h
                    out_bw = (new_x2 - new_x1) / target_w
                    out_bh = (new_y2 - new_y1) / target_h

                    if out_bw > 0.003 and out_bh > 0.003:
                        out_labels.append([cls_id, out_xc, out_yc, out_bw, out_bh])

        return out_img, np.array(out_labels, dtype=np.float32) if out_labels else np.zeros((0, 5), dtype=np.float32)


class Cutout(BaseAugmentation):
    """
    随机遮挡 - 模拟红外目标被遮挡

    航拍中目标可能被：云层、其他目标、建筑物遮挡。
    切掉一部分让模型学会用局部特征检测目标。对红外小目标效果显著。
    """
    def __init__(self, max_holes: int = 1, max_size: float = 0.1, p: float = 0.3):
        self.max_holes = max_holes
        self.max_size = max_size
        self.p = p

    def apply(self, image: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            h, w = image.shape[1], image.shape[2]
            for _ in range(random.randint(1, self.max_holes)):
                hole_w = int(w * random.uniform(0.02, self.max_size))
                hole_h = int(h * random.uniform(0.02, self.max_size))
                x = random.randint(0, w - hole_w)
                y = random.randint(0, h - hole_h)
                # 用图像均值填充（保持热辐射连续性）
                fill_value = int(np.mean(image))
                image[:, y:y+hole_h, x:x+hole_w] = fill_value
        return image, labels


class CopyPaste(BaseAugmentation):
    """
    Copy-Paste 增强 — 裁剪目标块并粘贴到另一图像位置

    对红外小目标特别有效：
    1. 缓解小目标正样本稀疏问题
    2. 目标被放在不同背景上下文中，学习丰富的特征
    3. 基于局部灰度统计匹配粘贴位置，保证物理合理性

    实现：
    - 从训练集中随机取另一张图（通过 dataset 引用）
    - 提取全部目标块（边界框 + 边缘扩展）
    - 在目标图像上搜索灰度近似的粘贴位置
    - Alpha 融合边界过渡
    """
    def __init__(self, p: float = 0.3, dataset=None, margin_ratio: float = 0.15,
                 search_samples: int = 20, alpha: float = 0.85):
        """
        Args:
            p: 应用概率
            dataset: 数据集引用，用于获取随机样本
            margin_ratio: 裁剪时在标注框外扩展的比例（相对短边）
            search_samples: 搜索粘贴位置的采样点数量
            alpha: Alpha 融合系数（0=完全透明, 1=完全不透明）
        """
        self.p = p
        self.dataset = dataset
        self.margin_ratio = margin_ratio
        self.search_samples = search_samples
        self.alpha = alpha

    def _extract_patches(self, image: np.ndarray, labels: np.ndarray
                         ) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
        """
        从图像中裁剪所有目标块。
        Returns: [(patch_img, label_xywh, gray_mean, gray_std), ...]
        其中 label_xywh = [cls_id, x_center, y_center, width, height]
        """
        patches = []
        h, w = image.shape[1], image.shape[2]  # [C, H, W]
        for label in labels:
            cls_id, xc, yc, bw, bh = label
            # 反归一化
            cx, cy = int(xc * w), int(yc * h)
            bw_px, bh_px = int(bw * w), int(bh * h)
            if bw_px < 3 or bh_px < 3:
                continue
            margin = max(1, int(min(bw_px, bh_px) * self.margin_ratio))
            x1 = max(0, cx - bw_px // 2 - margin)
            y1 = max(0, cy - bh_px // 2 - margin)
            x2 = min(w, cx + bw_px // 2 + margin)
            y2 = min(h, cy + bh_px // 2 + margin)
            patch = image[:, y1:y2, x1:x2].copy()
            # 计算局部灰度统计（取中心区域）
            center_region = image[0, max(0, cy-3):min(h, cy+3),
                                  max(0, cx-3):min(w, cx+3)]
            gray_mean = float(np.mean(center_region))
            gray_std = float(np.std(center_region)) + 1e-6
            patches.append((patch, label, gray_mean, gray_std))
        return patches

    def _find_best_location(self, image: np.ndarray, patch: np.ndarray,
                             target_gray_mean: float, target_gray_std: float
                             ) -> Optional[tuple[int, int, int, int]]:
        """
        在图像中搜索灰度近似的位置。
        Returns: (x1, y1, x2, y2) 粘贴位置
        """
        h_img, w_img = image.shape[1], image.shape[2]
        ph, pw = patch.shape[1], patch.shape[2]
        best_score = float('inf')
        best_bbox = None
        for _ in range(self.search_samples):
            x = random.randint(0, max(1, w_img - pw))
            y = random.randint(0, max(1, h_img - ph))
            region = image[0, y:min(h_img, y+ph), x:min(w_img, x+pw)]
            if region.size < 4:
                continue
            region_mean = float(np.mean(region))
            region_std = float(np.std(region)) + 1e-6
            # 灰度统计匹配得分
            mean_diff = abs(region_mean - target_gray_mean) / target_gray_std
            std_ratio = region_std / target_gray_std
            score = mean_diff + abs(1.0 - std_ratio)
            if score < best_score:
                best_score = score
                best_bbox = (x, y, x + pw, y + ph)
        return best_bbox

    def apply(self, image: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() >= self.p or self.dataset is None or len(labels) == 0:
            return image, labels

        h, w = image.shape[1], image.shape[2]
        # 从数据集中随机取另一张图（跳过增强，防止递归）
        ds_len = len(self.dataset)
        src_idx = random.randint(0, ds_len - 1)
        from .dataset import YOLODataset
        YOLODataset._recursive_guard = True
        try:
            src_img_t, src_labels_t, _ = self.dataset[src_idx]
        finally:
            YOLODataset._recursive_guard = False

        src_img = src_img_t.numpy() if hasattr(src_img_t, 'numpy') else np.array(src_img_t)
        src_labels = src_labels_t.numpy() if hasattr(src_labels_t, 'numpy') else np.array(src_labels_t)
        # dataset 返回的可能是 [C,H,W] 或 [H,W,C]；统一为 [C,H,W] uint8
        if src_img.ndim == 2:
            src_img = np.expand_dims(src_img, axis=0)
        elif src_img.ndim == 3 and src_img.shape[-1] in (1, 3):
            src_img = np.transpose(src_img, (2, 0, 1))
        src_img = src_img.astype(np.uint8)

        if not isinstance(src_labels, np.ndarray) or len(src_labels) == 0:
            return image, labels

        # 提取源图的目标块
        patches = self._extract_patches(src_img, src_labels)
        if not patches:
            return image, labels

        new_labels = labels.tolist() if isinstance(labels, np.ndarray) else labels.copy()
        out_img = image.copy()

        for patch, src_label, gray_mean, gray_std in patches:
            ph, pw = patch.shape[1], patch.shape[2]
            if ph < 4 or pw < 4:
                continue
            loc = self._find_best_location(out_img, patch, gray_mean, gray_std)
            if loc is None:
                continue
            x1, y1, x2, y2 = loc
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue

            # Alpha 融合
            if self.alpha < 1.0:
                out_img[:, y1:y2, x1:x2] = (out_img[:, y1:y2, x1:x2] * (1 - self.alpha) +
                                             patch * self.alpha).astype(np.uint8)
            else:
                out_img[:, y1:y2, x1:x2] = patch

            # 边缘 2px 软化（与原始背景融合）
            if ph > 4 and pw > 4:
                # Store original background for proper blending
                bg_original = image.copy()

                for c in range(out_img.shape[0]):
                    for dx in range(min(2, pw)):
                        fade = (dx + 1) / 3.0
                        out_img[c, y1:y2, x1+dx] = (
                            bg_original[c, y1:y2, x1+dx] * (1 - fade) +
                            out_img[c, y1:y2, x1+dx] * fade
                        ).astype(np.uint8)
                        out_img[c, y1:y2, x2-dx-1] = (
                            bg_original[c, y1:y2, x2-dx-1] * (1 - fade) +
                            out_img[c, y1:y2, x2-dx-1] * fade
                        ).astype(np.uint8)
                    # 垂直边缘淡化
                    for dy in range(min(2, ph)):
                        fade = (dy + 1) / 3.0
                        out_img[c, y1+dy, x1:x2] = (bg_original[c, y1+dy, x1:x2] * (1 - fade) +
                                                     out_img[c, y1+dy, x1:x2] * fade).astype(np.uint8)
                        out_img[c, y2-dy-1, x1:x2] = (bg_original[c, y2-dy-1, x1:x2] * (1 - fade) +
                                                       out_img[c, y2-dy-1, x1:x2] * fade).astype(np.uint8)

            # 计算粘贴后新的归一化坐标
            paste_cx = ((x1 + x2) / 2) / w
            paste_cy = ((y1 + y2) / 2) / h
            paste_bw = (x2 - x1) / w
            paste_bh = (y2 - y1) / h

            # 过滤：太小或越界的不加
            if paste_bw > 0.005 and paste_bh > 0.005 and paste_cx > 0 and paste_cy > 0:
                new_labels.append([src_label[0], paste_cx, paste_cy, paste_bw, paste_bh])

        return out_img, np.array(new_labels, dtype=np.float32) if new_labels else np.zeros((0, 5), dtype=np.float32)


class ComposeAugmentations:
    """
    组合多个数据增强 - 专为时序堆叠优化

    === 核心设计 ===
    对于时序堆叠图像 [3, H, W] (channels = t-1, t, t+1 三帧)：

    几何变换类（翻转/旋转/缩放/平移）：
        - 对3个通道统一执行
        - 原因：时序堆叠模拟的是同一场景的相邻帧，几何结构必须一致
        - 如果三个通道做不同的几何变换，t-1→t→t+1的运动模式被破坏

    光度变换类（亮度/对比度/噪声/模糊）：
        - 对3个通道独立随机执行
        - 原因：真实红外视频中帧间噪声、亮度波动是独立的
        - 33ms(30fps) 内传感器读出噪声、背景热辐射变化都可能导致帧间差异

    如果单通道输入：所有变换统一执行（等同于传统增强）。
    """

    _GEOMETRIC_OPS = ('randomhorizontalflip', 'randomverticalflip', 'randomrotate90',
                       'randomscale', 'randomtranslate', 'copypaste', 'mosaic')
    _PHOTOMETRIC_OPS = ('randombrightnesscontrast', 'randomnoise', 'randomblur', 'cutout')

    def __init__(self, augmentations: List[BaseAugmentation]):
        self.augmentations = augmentations

    def apply(self, image: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        num_channels = image.shape[0]

        for aug in self.augmentations:
            aug_name = type(aug).__name__.lower()

            if aug_name in self._GEOMETRIC_OPS:
                # ========== 几何变换：所有通道统一执行 ==========
                # 时序堆叠帧的几何结构必须一致，否则运动模式被破坏
                image, labels = aug.apply(image, labels)

            elif aug_name in self._PHOTOMETRIC_OPS and num_channels > 1:
                # ========== 光度变换：每个通道独立执行 ==========
                # 真实红外视频中帧间噪声、亮度变化是独立的
                processed_channels = []
                for c in range(num_channels):
                    single = image[c:c+1, :, :]  # [1, H, W]
                    single, _ = aug.apply(single, labels.copy())
                    processed_channels.append(single)
                image = np.concatenate(processed_channels, axis=0)
            else:
                # 单通道或非分类增强
                image, labels = aug.apply(image, labels)

        return image, labels


# ====================
# 推荐的增强组合
# ====================

def get_infrared_augmentation_pipeline(mode: str = 'medium', dataset=None,
                                        copy_paste_prob: float = 0.3) -> ComposeAugmentations:
    """
    返回针对红外图像优化的增强管道

    Args:
        mode: 'light' | 'medium' | 'heavy'
        dataset: 数据集引用（Mosaic/CopyPaste 需要随机采样）
        copy_paste_prob: Copy-Paste 增强概率（0=关闭）
        
    Returns:
        ComposeAugmentations 实例
    """
    if mode == 'light':
        return ComposeAugmentations([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.3),
            RandomBrightnessContrast(p=0.5, brightness_delta=0.2, contrast_delta=0.2),
        ])
    
    elif mode == 'medium':
        aug_list = [
            CopyPaste(p=copy_paste_prob, dataset=dataset),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomBrightnessContrast(p=0.7, brightness_delta=0.3, contrast_delta=0.3),
            RandomNoise(noise_type='gaussian', gaussian_std=0.03, p=0.3),
            RandomBlur(max_kernel=3, blur_type='gaussian', p=0.2),
            RandomTranslate(max_translate=0.1, p=0.3),
        ]
        return ComposeAugmentations(aug_list)
    
    else:  # 'heavy'
        aug_list = [
            CopyPaste(p=copy_paste_prob, dataset=dataset),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
        ]
        aug_list.extend([
            Mosaic(p=0.3, target_size=640, dataset=dataset),
            RandomRotate90(p=0.3),
            RandomScale(scale_range=(0.7, 1.3), p=0.3),
            RandomBrightnessContrast(p=0.8, brightness_delta=0.4, contrast_delta=0.4),
            RandomNoise(noise_type='gaussian', gaussian_std=0.05, p=0.4),
            RandomBlur(max_kernel=5, blur_type='gaussian', p=0.3),
            RandomTranslate(max_translate=0.1, p=0.3),
            Cutout(max_holes=1, max_size=0.08, p=0.2),
        ])
        return ComposeAugmentations(aug_list)