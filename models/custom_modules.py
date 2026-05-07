"""
Custom modules for IR-YOLOv8n: ECA, GSConv, C2f_GS

Usage:
    from models.custom_modules import register_custom_modules
    register_custom_modules()   # call before creating a YOLO model from YAML
"""
import math

import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv, Bottleneck


class ECA(nn.Module):
    """Efficient Channel Attention — 1D conv based, no dimension reduction"""
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        k = int(abs(math.log2(channels) / gamma + b / gamma))
        k = k if k % 2 else k + 1  # ensure odd kernel size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y).view(b, c, 1, 1)
        return x * self.sigmoid(y)


class GSConv(nn.Module):
    """GSConv: standard Conv → DWConv → Concat → Channel Shuffle"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_half = c2 // 2
        self.conv = nn.Conv2d(c1, c_half, k, s, p or k // 2, groups=g, dilation=d, bias=False)
        self.bn1 = nn.BatchNorm2d(c_half)
        self.dwconv = nn.Conv2d(c_half, c_half, 5, 1, 2, groups=c_half, bias=False)
        self.bn2 = nn.BatchNorm2d(c_half)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        x1 = self.act(self.bn1(self.conv(x)))
        x2 = self.act(self.bn2(self.dwconv(x1)))
        out = torch.cat([x1, x2], dim=1)
        # Channel Shuffle: (b, 2*c_half, h, w) → (b, 2, c_half, h, w) → transpose → (b, c_half, 2, h, w)
        b, c, h, w = out.size()
        out = out.view(b, 2, c // 2, h, w).transpose(1, 2).contiguous().view(b, c, h, w)
        return out


class Bottleneck_GS(nn.Module):
    """Bottleneck with GSConv"""
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = GSConv(c1, c_, 1, 1)
        self.cv2 = GSConv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f_GS(nn.Module):
    """C2f with GSConv bottlenecks — lightweight neck building block"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = GSConv(c1, 2 * self.c, 1, 1)
        self.cv2 = GSConv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck_GS(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ========== Registration API ==========

import inspect
import re

_registered = False


def register_custom_modules():
    """Register ECA, GSConv, C2f_GS with ultralytics for YAML model definitions.

    Must be called before YOLO("path/to/yaml") if the YAML references these modules.
    Safe to call multiple times (idempotent).
    """
    global _registered
    if _registered:
        return
    _registered = True

    import ultralytics.nn.tasks as tasks

    tasks.ECA = ECA
    tasks.GSConv = GSConv
    tasks.Bottleneck_GS = Bottleneck_GS
    tasks.C2f_GS = C2f_GS

    original_parse_model = tasks.parse_model
    patched = _build_patched_parse_model(original_parse_model)
    tasks.parse_model = patched


def _build_patched_parse_model(original):
    """Build a patched parse_model by injecting custom modules into its frozensets."""
    try:
        src = inspect.getsource(original)
    except OSError:
        return original

    src = _inject_into_frozenset(src, "base_modules", "GSConv, C2f_GS")
    src = _inject_into_frozenset(src, "repeat_modules", "C2f_GS")

    import ultralytics.nn.tasks as tasks
    namespace: dict = {}
    exec(src, tasks.__dict__, namespace)
    return namespace.get("parse_model", original)


def _inject_into_frozenset(source: str, varname: str, modules: str) -> str:
    """Insert `modules` before the closing }} of frozenset named `varname`."""
    pattern = rf'({varname}\s*=\s*frozenset\s*\(\s*\{{[^}}]*)'
    match = re.search(pattern, source, re.DOTALL)
    if not match:
        return source
    end = match.end()
    return source[:end] + " " + modules + "," + source[end:]
