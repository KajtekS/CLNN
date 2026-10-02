import cv2
from PREPROC.preproc import Preproc

class Pipeline:
    @staticmethod
    def run(config: dict) -> None:
        print(config)
        preproc = Preproc()
        preproc.proces(cv2.imread("DATA/IMG_2998.png"))