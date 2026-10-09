import numpy as np
import pytest
import yaml

from TOOLS.resampling import video_resampler, gt_resampler
from TOOLS.hash import hash_yaml

# TESTING RESAMPLERS
# region


@pytest.mark.parametrize(
    "fs_start, fs_end, expected_frames",
    [
        (60, 30, 50),
        (30, 60, 200),
        (60, 20, 34),
    ],
)
def test_resampling_video(fs_start, fs_end, expected_frames):
    frames = np.ones((100, 72, 72, 3), dtype=np.float32)

    resampled = video_resampler(frames, fs_start, fs_end)

    assert len(resampled) == expected_frames
    assert resampled.shape[1:] == (72, 72, 3)
    assert np.isfinite(resampled).all()


@pytest.mark.parametrize(
    "fs_start, fs_end, expected_gt",
    [
        (60, 30, 50),
        (30, 60, 200),
        (60, 20, 34),
    ],
)
def test_resampled_gt(fs_start, fs_end, expected_gt):
    gt = np.ones((100), dtype=np.float32)

    resampled = gt_resampler(gt, fs_start, fs_end)

    assert len(resampled) == expected_gt
    assert resampled.shape == (expected_gt,)
    assert np.isfinite(resampled).all()


# endregion


# Testing Hash functions
# region
def test_hash_yaml(tmp_path):
    data_input = {"name": "john", "surname": "doe"}
    data_input_2 = {"surname": "doe", "name": "john"}
    data_path = tmp_path / "data.yaml"
    data_path_2 = tmp_path / "data2.yaml"

    with open(data_path, "w") as f:
        yaml.dump(data_input, f)
    with open(data_path_2, "w") as f:
        yaml.dump(data_input_2, f)

    hash_1 = hash_yaml(data_path)
    hash_2 = hash_yaml(data_path_2)

    assert hash_1 == hash_2


# endregion
