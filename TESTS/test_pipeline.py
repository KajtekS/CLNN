from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from PIPELINE.piper import Piper


def _create_dummy_avi(path: Path, num_frames: int = 6, width: int = 10, height: int = 10) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), ((i % 12) + 1) * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


# Testing raw and matrix loaders, and stage saving
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
    data = [(np.ones((3, 3, 3)), np.zeros(3)) for _ in range(3)]
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


def test_raw_loader(tmp_path):
    pipe = Piper(Path("."))
    subj = tmp_path / "subject1"
    subj.mkdir()
    _create_dummy_avi(subj / "vid.avi", num_frames=6)
    (subj / "ground_truth.txt").write_text("1.0 2.0 3.0 4.0 5.0 6.0\n", encoding="utf-8")

    video_out, gt_out = pipe.raw_loader("UBFC-RPPG", tmp_path)

    assert len(video_out) == 1
    assert video_out[0].shape == (6, 10, 10, 3)
    assert len(gt_out) == 1
    np.testing.assert_array_equal(gt_out[0], np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


# endregion


# Testing bronze stage
# region


def test_bronze_resampling():
    pipe = Piper(Path("."))
    data = np.ones((100, 72, 72, 3), dtype=np.float32)
    gt = np.ones(100, dtype=np.float32)

    resampled_data, resampled_gt = pipe.bronze(
        data, gt, fs_start=60, fs_gt_start=60, fs_end=30
    )

    assert len(resampled_data) == 50
    assert resampled_data.shape[1:] == (72, 72, 3)
    assert np.isfinite(resampled_data).all()

    assert len(resampled_gt) == 50
    assert resampled_gt.shape == (50,)
    assert np.isfinite(resampled_gt).all()


def test_bronze_different_gt_frequency():
    pipe = Piper(Path("."))
    data = np.ones((100, 32, 32, 3), dtype=np.float32)
    gt = np.ones(50, dtype=np.float32)

    resampled_data, resampled_gt = pipe.bronze(
        data, gt, fs_start=60, fs_gt_start=30, fs_end=30
    )

    assert len(resampled_data) == 50
    assert len(resampled_gt) == 50


# endregion


# Testing silver stage
# region


def test_silver_fallback_zeros():
    pipe = Piper(Path("."))
    frames = np.zeros((3, 100, 100, 3), dtype=np.uint8)
    out = pipe.silver(frames)

    assert out.shape == (3, 256, 256, 3)
    assert out.dtype == np.float32
    # Since all fallback frames are zeros, diff across all frames is zeros
    np.testing.assert_array_equal(out, np.zeros((3, 256, 256, 3), dtype=np.float32))


def test_silver_with_detected_face():
    pipe = Piper(Path("."))
    dummy_roi_1 = np.full((256, 256, 3), 100, dtype=np.uint8)
    dummy_roi_2 = np.full((256, 256, 3), 200, dtype=np.uint8)

    mock_extractor = MagicMock()
    mock_extractor.detect_landmarks.return_value = MagicMock()
    mock_extractor.extract_face_roi.side_effect = [dummy_roi_1, dummy_roi_2]
    pipe.extractor = mock_extractor

    frames = np.zeros((2, 100, 100, 3), dtype=np.uint8)
    out = pipe.silver(frames)

    assert out.shape == (2, 256, 256, 3)
    assert out.dtype == np.float32
    # First frame diff with itself is zero
    np.testing.assert_allclose(out[0], np.zeros((256, 256, 3), dtype=np.float32))
    # Second frame diff is 200/255.0 - 100/255.0 = 100/255.0
    expected_diff = np.full((256, 256, 3), 100.0 / 255.0, dtype=np.float32)
    np.testing.assert_allclose(out[1], expected_diff, rtol=1e-5)


# endregion


# Testing gold stage
# region


def test_gold_chunking():
    pipe = Piper(Path("."))
    # 70 frames, chunk_size=20 -> 3 chunks of 20, 10 frames dropped
    data = np.arange(70 * 2 * 2 * 3).reshape(70, 2, 2, 3).astype(np.float32)
    gt = np.arange(70).astype(np.float32)

    chunked_data, chunked_gt = pipe.gold(data, gt, chunk_size=20)

    assert chunked_data.shape == (3, 20, 2, 2, 3)
    assert chunked_gt.shape == (3, 20)

    np.testing.assert_array_equal(chunked_data[0], data[0:20])
    np.testing.assert_array_equal(chunked_data[1], data[20:40])
    np.testing.assert_array_equal(chunked_data[2], data[40:60])

    np.testing.assert_array_equal(chunked_gt[0], gt[0:20])
    np.testing.assert_array_equal(chunked_gt[1], gt[20:40])
    np.testing.assert_array_equal(chunked_gt[2], gt[40:60])


def test_gold_exact_multiple():
    pipe = Piper(Path("."))
    data = np.ones((60, 4, 4, 3), dtype=np.float32)
    gt = np.ones(60, dtype=np.float32)

    chunked_data, chunked_gt = pipe.gold(data, gt, chunk_size=30)

    assert chunked_data.shape == (2, 30, 4, 4, 3)
    assert chunked_gt.shape == (2, 30)


# endregion


# Testing full pipeline end-to-end
# region


def test_full_pipeline_end_to_end(tmp_path):
    pipe = Piper(Path("."))

    # 1. RAW stage: create dataset structure and load with raw_loader
    raw_dir = tmp_path / "raw"
    subj = raw_dir / "subject1"
    subj.mkdir(parents=True)
    _create_dummy_avi(subj / "vid.avi", num_frames=60)
    gt_signal = " ".join(str(float(i)) for i in range(60)) + "\n"
    (subj / "ground_truth.txt").write_text(gt_signal, encoding="utf-8")

    raw_videos, raw_gts = pipe.raw_loader("UBFC-RPPG", raw_dir)
    assert len(raw_videos) == 1
    assert raw_videos[0].shape == (60, 10, 10, 3)
    assert len(raw_gts) == 1
    assert raw_gts[0].shape == (60,)

    # 2. BRONZE stage: resample 60 fps -> 30 fps, save and reload
    stages_root = tmp_path / "stages"
    bronze_video, bronze_gt = pipe.bronze(
        raw_videos[0], raw_gts[0], fs_start=60, fs_gt_start=60, fs_end=30
    )
    assert bronze_video.shape == (30, 10, 10, 3)
    assert bronze_gt.shape == (30,)

    pipe.save_stage([(bronze_video, bronze_gt)], stages_root, "hash_bronze", "BRONZE")
    loaded_bronze_video, loaded_bronze_gt = pipe.matrix_loader(
        stages_root / "BRONZE" / "hash_bronze"
    )
    assert len(loaded_bronze_video) == 1
    np.testing.assert_array_equal(loaded_bronze_video[0], bronze_video)
    np.testing.assert_array_equal(loaded_bronze_gt[0], bronze_gt)

    # 3. SILVER stage: ROI extraction, normalization and differences, save and reload
    silver_video = pipe.silver(loaded_bronze_video[0])
    assert silver_video.shape == (30, 256, 256, 3)
    assert silver_video.dtype == np.float32

    pipe.save_stage([(silver_video, loaded_bronze_gt[0])], stages_root, "hash_silver", "SILVER")
    loaded_silver_video, loaded_silver_gt = pipe.matrix_loader(
        stages_root / "SILVER" / "hash_silver"
    )
    assert len(loaded_silver_video) == 1
    np.testing.assert_array_equal(loaded_silver_video[0], silver_video)
    np.testing.assert_array_equal(loaded_silver_gt[0], loaded_bronze_gt[0])

    # 4. GOLD stage: chunk into batches for training, save and reload
    chunk_size = 10
    gold_video, gold_gt = pipe.gold(
        loaded_silver_video[0], loaded_silver_gt[0], chunk_size=chunk_size
    )
    assert gold_video.shape == (3, 10, 256, 256, 3)
    assert gold_gt.shape == (3, 10)

    pipe.save_stage([(gold_video, gold_gt)], stages_root, "hash_gold", "GOLD")
    loaded_gold_video, loaded_gold_gt = pipe.matrix_loader(
        stages_root / "GOLD" / "hash_gold"
    )
    assert len(loaded_gold_video) == 1
    np.testing.assert_array_equal(loaded_gold_video[0], gold_video)
    np.testing.assert_array_equal(loaded_gold_gt[0], gold_gt)


# endregion
