from pathlib import Path
import csv
import json

import cv2
import numpy as np
import pandas as pd
import pytest

from LOADER.loader import (
    PureLoader,
    SumsLoader,
    UbfcPhysLoader,
    UbfcRppgLoader,
    get_loader,
)
from LOADER.readers import (
    PureReader,
    SumsReader,
    UbfcPhysReader,
    UbfcRppgReader,
    get_dataset_reader,
)
from LOADER.vid_readers import AviReader, PngReader, get_reader


def _create_dummy_avi(
    path: Path, num_frames: int = 3, width: int = 10, height: int = 10
) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), (i + 1) * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def _create_dummy_pngs(
    folder: Path, num_frames: int = 3, width: int = 10, height: int = 10
) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(num_frames):
        frame = np.full((height, width, 3), (i + 1) * 10, dtype=np.uint8)
        cv2.imwrite(str(folder / f"frame_{i:03d}.png"), frame)
    return folder


# Testing loaders
# region


@pytest.mark.parametrize(
    "name, expected_class",
    [
        ("SUMS", SumsLoader),
        ("sums", SumsLoader),
        ("UBFC-PHYS", UbfcPhysLoader),
        ("ubfc-phys", UbfcPhysLoader),
        ("UBFC-RPPG", UbfcRppgLoader),
        ("PURE", PureLoader),
        ("pure", PureLoader),
    ],
)
def test_get_loader(name, expected_class):
    assert get_loader(name) is expected_class


def test_get_loader_invalid():
    with pytest.raises(ValueError, match="Unsupported dataset"):
        get_loader("UNKNOWN")


def test_sums_loader(tmp_path):
    task_dir = tmp_path / "060200" / "v01"
    task_dir.mkdir(parents=True)
    video_file = task_dir / "video_ZIP_H264_face.avi"
    video_file.touch()
    bvp_file = task_dir / "BVP.csv"
    bvp_file.touch()

    # Files that should be ignored
    (tmp_path / "060200" / "ignored.txt").touch()
    invalid_task = tmp_path / "060200" / "v02"
    invalid_task.mkdir()
    (invalid_task / "other.avi").touch()

    result = SumsLoader.load_data(tmp_path)

    assert len(result) == 1
    assert result[0]["subject"] == "060200_v01"
    assert result[0]["video_path"] == video_file
    assert result[0]["gt_path"] == bvp_file


def test_ubfc_phys_loader(tmp_path):
    subj_dir = tmp_path / "s1"
    subj_dir.mkdir()

    vid1 = subj_dir / "vid_s1_T1.avi"
    vid1.touch()
    gt1 = subj_dir / "bvp_s1_T1.csv"
    gt1.touch()

    # Video without matching ground truth
    vid2 = subj_dir / "vid_s1_T2.avi"
    vid2.touch()

    result = UbfcPhysLoader.load_data(tmp_path)

    assert len(result) == 1
    assert result[0]["subject"] == "s1_T1"
    assert result[0]["video_path"] == vid1
    assert result[0]["gt_path"] == gt1


def test_ubfc_rppg_loader(tmp_path):
    subj_dir = tmp_path / "subject1"
    subj_dir.mkdir()
    vid = subj_dir / "vid.avi"
    vid.touch()
    gt = subj_dir / "ground_truth.txt"
    gt.touch()

    # Incomplete subject directory
    empty_subj = tmp_path / "subject2"
    empty_subj.mkdir()
    (empty_subj / "vid.avi").touch()

    result = UbfcRppgLoader.load_data(tmp_path)

    assert len(result) == 1
    assert result[0]["subject"] == "subject1"
    assert result[0]["video_path"] == vid
    assert result[0]["gt_path"] == gt


def test_pure_loader(tmp_path):
    subj_dir = tmp_path / "01-01"
    subj_dir.mkdir()
    vid_folder = subj_dir / "01-01"
    vid_folder.mkdir()
    gt = subj_dir / "01-01.json"
    gt.touch()

    # Incomplete subject directory
    other_dir = tmp_path / "02-01"
    other_dir.mkdir()

    result = PureLoader.load_data(tmp_path)

    assert len(result) == 1
    assert result[0]["subject"] == "01-01"
    assert result[0]["video_path"] == vid_folder
    assert result[0]["gt_path"] == gt


# endregion


# Testing vid_readers
# region


@pytest.mark.parametrize(
    "video_type, expected_class",
    [
        ("AVI", AviReader),
        ("avi", AviReader),
        ("PNG", PngReader),
        ("png", PngReader),
    ],
)
def test_get_reader(video_type, expected_class):
    assert get_reader(video_type) is expected_class


def test_get_reader_invalid():
    with pytest.raises(ValueError, match="Unsupported reader filetype"):
        get_reader("MP4")


def test_avi_reader(tmp_path):
    avi_path = _create_dummy_avi(tmp_path / "test.avi", num_frames=3)
    frames = AviReader.read_video(avi_path)

    assert isinstance(frames, np.ndarray)
    assert frames.shape == (3, 10, 10, 3)


def test_avi_reader_nonexistent(tmp_path):
    with pytest.raises(ValueError, match="Could not open video"):
        AviReader.read_video(tmp_path / "nonexistent.avi")


def test_avi_reader_empty_file(tmp_path):
    empty_file = tmp_path / "empty.avi"
    empty_file.touch()
    with pytest.raises(ValueError):
        AviReader.read_video(empty_file)


def test_png_reader(tmp_path):
    png_folder = _create_dummy_pngs(tmp_path / "frames", num_frames=3)
    frames = PngReader.read_video(png_folder)

    assert isinstance(frames, np.ndarray)
    assert frames.shape == (3, 10, 10, 3)


def test_png_reader_not_a_directory(tmp_path):
    file_path = tmp_path / "file.txt"
    file_path.touch()
    with pytest.raises(ValueError, match="Expected a directory"):
        PngReader.read_video(file_path)


def test_png_reader_empty_directory(tmp_path):
    empty_folder = tmp_path / "empty_dir"
    empty_folder.mkdir()
    with pytest.raises(ValueError, match="No PNG files found"):
        PngReader.read_video(empty_folder)


def test_png_reader_different_dimensions(tmp_path):
    folder = tmp_path / "mixed_frames"
    folder.mkdir()
    img1 = np.zeros((10, 10, 3), dtype=np.uint8)
    img2 = np.zeros((20, 20, 3), dtype=np.uint8)
    cv2.imwrite(str(folder / "frame_0.png"), img1)
    cv2.imwrite(str(folder / "frame_1.png"), img2)

    with pytest.raises(ValueError, match="PNG frames have different dimensions"):
        PngReader.read_video(folder)


# endregion


# Testing readers
# region


@pytest.mark.parametrize(
    "name, expected_class",
    [
        ("SUMS", SumsReader),
        ("sums", SumsReader),
        ("UBFC-PHYS", UbfcPhysReader),
        ("ubfc-phys", UbfcPhysReader),
        ("UBFC-RPPG", UbfcRppgReader),
        ("PURE", PureReader),
        ("pure", PureReader),
    ],
)
def test_get_dataset_reader(name, expected_class):
    assert get_dataset_reader(name) is expected_class


def test_get_dataset_reader_invalid():
    with pytest.raises(ValueError, match="Unsupported dataset"):
        get_dataset_reader("UNKNOWN")


def test_sums_reader(tmp_path):
    video_path = _create_dummy_avi(tmp_path / "video.avi", num_frames=3)
    gt_path = tmp_path / "BVP.csv"
    pd.DataFrame({"bvp": [0.5, 1.0, 1.5]}).to_csv(gt_path, index=False)

    video, gt = SumsReader.read(video_path, gt_path)

    assert video.shape == (3, 10, 10, 3)
    np.testing.assert_array_equal(gt, np.array([0.5, 1.0, 1.5], dtype=np.float64))


def test_ubfc_phys_reader(tmp_path):
    video_path = _create_dummy_avi(tmp_path / "video.avi", num_frames=3)
    gt_path = tmp_path / "bvp.csv"
    with open(gt_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([10.5])
        writer.writerow([20.0])
        writer.writerow([30.5])

    video, gt = UbfcPhysReader.read(video_path, gt_path)

    assert video.shape == (3, 10, 10, 3)
    np.testing.assert_array_equal(gt, np.array([10.5, 20.0, 30.5], dtype=np.float64))


def test_ubfc_rppg_reader(tmp_path):
    video_path = _create_dummy_avi(tmp_path / "video.avi", num_frames=3)
    gt_path = tmp_path / "ground_truth.txt"
    gt_path.write_text("1.1 2.2 3.3\nsecond line to ignore\n", encoding="utf-8")

    video, gt = UbfcRppgReader.read(video_path, gt_path)

    assert video.shape == (3, 10, 10, 3)
    np.testing.assert_array_equal(gt, np.array([1.1, 2.2, 3.3], dtype=np.float64))


def test_pure_reader(tmp_path):
    video_path = _create_dummy_pngs(tmp_path / "frames", num_frames=3)
    gt_path = tmp_path / "gt.json"
    gt_data = {
        "/FullPackage": [
            {"Value": {"waveform": 100.0}},
            {"Value": {"waveform": 200.0}},
            {"Value": {"waveform": 300.0}},
        ]
    }
    gt_path.write_text(json.dumps(gt_data), encoding="utf-8")

    video, gt = PureReader.read(video_path, gt_path)

    assert video.shape == (3, 10, 10, 3)
    np.testing.assert_array_equal(gt, np.array([100.0, 200.0, 300.0], dtype=np.float64))


# endregion
