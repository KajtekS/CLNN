from PIPELINE.piper import Piper
from pathlib import Path
import numpy as np

# Testing bronze_loader
# region


def test_bronze_loader(tmp_path):
    folder = tmp_path / "subject1"
    folder.mkdir()

    video1 = np.array([1, 2, 3])
    gt1 = np.array([3, 2, 1])

    np.save(folder / "video.npy", video1)
    np.save(folder / "gt.npy", gt1)

    folder = tmp_path / "subject2"
    folder.mkdir()

    video2 = np.array([4, 5, 6])
    gt2 = np.array([6, 5, 4])

    np.save(folder / "video.npy", video2)
    np.save(folder / "gt.npy", gt2)

    pipe = Piper(Path("."))

    video_out, gt_out = pipe.bronze_loader(tmp_path)

    assert any(np.array_equal(x, video1) for x in video_out)
    assert any(np.array_equal(x, video2) for x in video_out)

    assert any(np.array_equal(x, gt1) for x in gt_out)
    assert any(np.array_equal(x, gt2) for x in gt_out)


# endregion
