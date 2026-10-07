from scipy.signal import resample_poly

def video_resampler(video, fs_start, fs_end):
    resampled = resample_poly(video, fs_end, fs_start, axis=0)
    return resampled

def gt_resampler(signal, fs_start, fs_end):
    resampled = resample_poly(signal, fs_end, fs_start)
    return resampled