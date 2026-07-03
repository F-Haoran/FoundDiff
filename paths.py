"""Local data paths for FoundDiff (under the project data/mayo folder)."""
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Root for all Mayo-related data inside FoundDiff
MAYO_DATA_ROOT = os.path.join(_PROJECT_ROOT, 'data', 'mayo')

# Raw DICOM from Download_Mayo.py
MAYO_DICOM_ROOT = os.path.join(MAYO_DATA_ROOT, 'dicom')

# Preprocessed 2D .npy slices (expected layout from the original authors)
MAYO2020_AB = os.path.join(MAYO_DATA_ROOT, 'Mayo2020_ab_2d')
MAYO2020_LUNG = os.path.join(MAYO_DATA_ROOT, 'Mayo2020_lung_2d')
MAYO2020_HEAD = os.path.join(MAYO_DATA_ROOT, 'Mayo2020_head_2d_2')
MAYO2020_HEAD_V1 = os.path.join(MAYO_DATA_ROOT, 'Mayo2020_head_2d')
MAYO2016 = os.path.join(MAYO_DATA_ROOT, 'Mayo2016_2d')
CQ500 = os.path.join(MAYO_DATA_ROOT, 'CQ500_2d')

# External open CT (nii.gz -> Preprocess_nifti.py)
EXTERNAL_CT = os.path.join(_PROJECT_ROOT, 'data', 'external', 'external_2d')
EXTERNAL_NIFTI = os.path.join(_PROJECT_ROOT, 'data', 'external', 'nifti')

# Custom dataset: {CODE}_CT.nii.gz (noisy input) or {CODE}_LDCT.nii.gz (Mayo-style)
CUSTOM_NIFTI = os.path.join(_PROJECT_ROOT, 'data', 'custom', 'nifti')
CUSTOM_2D = os.path.join(_PROJECT_ROOT, 'data', 'custom', 'custom_2d')


def mayo_glob(base, phase, subfolder):
    """Glob pattern for a dose/subfolder under a Mayo2020 split."""
    return os.path.join(base, phase, subfolder, '*')
