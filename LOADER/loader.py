import os
import glob
from abc import ABC, abstractmethod


class loader(ABC):
    @abstractmethod
    @staticmethod
    def load_data(path):
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
        subject_dirs = glob.glob(os.path.join(path, '0602*'))
        
        for subj_dir in subject_dirs:
            subject_id = os.path.basename(subj_dir)
            
            for task in os.listdir(subj_dir):
                task_dir = os.path.join(subj_dir, task)
                if not os.path.isdir(task_dir):
                    continue
                
                vid_files = glob.glob(os.path.join(task_dir, '*face.avi'))
                bvp_file = os.path.join(task_dir, 'BVP.csv')
                
                if vid_files and os.path.exists(bvp_file):
                    data.append({
                        'subject': f"{subject_id}_{task}",
                        'video_path': vid_files[0],
                        'gt_path': bvp_file
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
        vid_files = glob.glob(os.path.join(path, "s*", "vid_*.avi"))
        
        for vid_path in vid_files:
            dir_name = os.path.dirname(vid_path)
            file_name = os.path.basename(vid_path)
            
            index = file_name.replace("vid_", "").replace(".avi", "")
            
            gt_path = os.path.join(dir_name, f"bvp_{index}.csv")
            
            if os.path.exists(gt_path):
                data.append({
                    'subject': index,
                    'video_path': vid_path,
                    'gt_path': gt_path
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
        subject_dirs = glob.glob(os.path.join(path, "subject*"))
        
        for subj_dir in subject_dirs:
            vid_path = os.path.join(subj_dir, "vid.avi")
            gt_path = os.path.join(subj_dir, "ground_truth.txt")
            
            if os.path.exists(vid_path) and os.path.exists(gt_path):
                data.append({
                    'subject': os.path.basename(subj_dir),
                    'video_path': vid_path,
                    'gt_path': gt_path
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
        subject_dirs = glob.glob(os.path.join(path, "*-*"))
        
        for subj_dir in subject_dirs:
            filename = os.path.basename(subj_dir)
            
            vid_path = os.path.join(subj_dir, filename) 
            gt_path = os.path.join(subj_dir, f"{filename}.json")
            
            if os.path.exists(vid_path) and os.path.exists(gt_path):
                data.append({
                    'subject': filename,
                    'video_path': vid_path,
                    'gt_path': gt_path
                })
        return data


def get_loader(dataset_name: str) -> loader:
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