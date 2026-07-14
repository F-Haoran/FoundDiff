"""HU <-> model [0,1] intensity mapping.

Input domain:  HU in [-2000, +1000]
Model domain:  norm in [0, 1]
Output domain: HU in [-2000, +1000]  (same as input)

Forward:  norm = clip((hu + 2000) / 3000, 0, 1)
Inverse:  hu   = clip(norm * 3000 - 2000, -2000, 1000)
"""

import numpy as np
import torch

HU_MIN = -2000
HU_MAX = 1000
HU_RANGE = HU_MAX - HU_MIN  # 3000


def hu_to_model_norm(hu):
    """[-2000, 1000] HU -> [0, 1]"""
    return np.clip((np.asarray(hu, dtype=np.float32) - HU_MIN) / HU_RANGE, 0.0, 1.0)


def model_norm_to_hu(norm):
    """[0, 1] -> [-2000, 1000] HU"""
    hu = np.asarray(norm, dtype=np.float32) * HU_RANGE + HU_MIN
    return np.clip(hu, HU_MIN, HU_MAX)


def preprocess_hu_slice(hu_slice):
    norm = hu_to_model_norm(hu_slice)
    tensor = torch.from_numpy(norm.astype(np.float32))
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor.unsqueeze(0)


def make_hu_nifti_header(ref_header):
    """Reset slope/intercept so ITK-SNAP shows the same HU as numpy (no double rescale)."""
    header = ref_header.copy()
    header['scl_slope'] = 1.0
    header['scl_inter'] = 0.0
    header.set_data_dtype(np.float32)
    header['cal_min'] = float(HU_MIN)
    header['cal_max'] = float(HU_MAX)
    return header
