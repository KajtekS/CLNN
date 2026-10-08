import numpy as np
from scipy.signal import resample_poly


def video_resampler(video: np.ndarray, fs_start: int, fs_end: int) -> np.ndarray:
    resampled = resample_poly(video, fs_end, fs_start, axis=0)
    return resampled


def gt_resampler(signal: np.ndarray, fs_start: int, fs_end: int) -> np.ndarray:
    resampled = resample_poly(signal, fs_end, fs_start)
    return resampled
