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

    video_out, gt_out = pipe.matrix_loader(tmp_path)

    assert any(np.array_equal(x, video1) for x in video_out)
    assert any(np.array_equal(x, video2) for x in video_out)

    assert any(np.array_equal(x, gt1) for x in gt_out)
    assert any(np.array_equal(x, gt2) for x in gt_out)

def test_save_stage(tmp_path):
    data = [(np.ones((3,3,3)), np.zeros(3)) for _ in range(3)]
    pipe = Piper(Path("."))
    pipe.save_stage(data, tmp_path, "0x64", "BRONZE")

    base_path = tmp_path / "BRONZE" / "0x64"

    for i in range(3):
        subject_path = base_path / f"subject_{i}"

        assert subject_path.exists()
        assert (subject_path / "video.npy").exists()
        assert (subject_path / "gt.npy").exists()

        matrix = np.load(subject_path / "video.npy")
        gt = np.load(subject_path / "gt.npy")

        np.testing.assert_array_equal(matrix, data[i][0])
        np.testing.assert_array_equal(gt, data[i][1])

# endregion
