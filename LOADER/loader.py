from pathlib import Path
from abc import ABC, abstractmethod


class loader(ABC):
    @abstractmethod
    @staticmethod
    def load_data(path) -> list[dict[str, str]]:
        """
        Returns a list of dictionaries with standardized keys:
        - 'subject': patient / experiment ID
        - 'video_path': path to the video file (or folder containing frames)
        - 'gt_path': path to the ground truth file (BVP/HR)
        """
        pass


class SumsLoader(loader):
    @staticmethod
    def load_data(path):
        """
        SUMS Structure:
        path/060200/v01/video_ZIP_H264_face.avi
        path/060200/v01/BVP.csv
        """
        data = []
        subject_dirs = list(Path(path).glob('0602*'))

        for subj_dir in subject_dirs:
            subject_id = subj_dir.name

            for task in subj_dir.iterdir():
                task_dir = task
                if not task_dir.is_dir():
                    continue

                vid_files = list(task_dir.glob('*face.avi'))
                bvp_file = task_dir / 'BVP.csv'

                if vid_files and bvp_file.exists():
                    data.append({
                        'subject': f"{subject_id}_{task.name}",
                        'video_path': str(vid_files[0]),
                        'gt_path': str(bvp_file)
                    })
        return data


class UbfcPhysLoader(loader):
    @staticmethod
    def load_data(path):
        """
        UBFC-PHYS Structure:
        path/s1/vid_s1_T1.avi
        path/s1/bvp_s1_T1.csv
        """
        data = []
        vid_files = list(Path(path).glob("s*/vid_*.avi"))

        for vid_path in vid_files:
            dir_name = vid_path.parent
            file_name = vid_path.name

            index = file_name.replace("vid_", "").replace(".avi", "")

            gt_path = dir_name / f"bvp_{index}.csv"

            if gt_path.exists():
                data.append({
                    'subject': index,
                    'video_path': str(vid_path),
                    'gt_path': str(gt_path)
                })
        return data


class UbfcRppgLoader(loader):
    @staticmethod
    def load_data(path):
        """
        UBFC-rPPG Structure:
        path/subject1/vid.avi
        path/subject1/ground_truth.txt
        """
        data = []
        subject_dirs = list(Path(path).glob("subject*"))

        for subj_dir in subject_dirs:
            vid_path = subj_dir / "vid.avi"
            gt_path = subj_dir / "ground_truth.txt"

            if vid_path.exists() and gt_path.exists():
                data.append({
                    'subject': subj_dir.name,
                    'video_path': str(vid_path),
                    'gt_path': str(gt_path)
                })
        return data


class PureLoader(loader):
    @staticmethod
    def load_data(path):
        """
        PURE Structure:
        path/01-01/01-01/ (folder containing .png files)
        path/01-01/01-01.json
        """
        data = []
        subject_dirs = list(Path(path).glob("*-*"))

        for subj_dir in subject_dirs:
            filename = subj_dir.name

            vid_path = subj_dir / filename
            gt_path = subj_dir / f"{filename}.json"

            if vid_path.exists() and gt_path.exists():
                data.append({
                    'subject': filename,
                    'video_path': str(vid_path),
                    'gt_path': str(gt_path)
                })
        return data


def get_loader(dataset_name: str) -> type[loader]:
    """Factory method to return the appropriate loader based on the config."""
    loaders = {
        'SUMS': SumsLoader,
        'UBFC-PHYS': UbfcPhysLoader,
        'UBFC-RPPG': UbfcRppgLoader,
        'PURE': PureLoader
    }
    dataset_name = dataset_name.upper()
    if dataset_name not in loaders:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Available: {list(loaders.keys())}")
    
    return loaders[dataset_name]