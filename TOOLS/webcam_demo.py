import cv2

from PROCESING.preproc import Preproc
from TOOLS.face_roi_extractor import FaceRoiExtractor



if __name__ == "__main__":
    frame_count = 0
    face_roi_extractor = FaceRoiExtractor("TOOLS/mediapipe/face_landmarker.task")
    preprocessor = Preproc()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    bbox = None
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            frame_bgr = frame.copy()
            face_bgr = None
            if frame_count >= 30:
                # bbox = preprocessor.detect_face(frame_bgr)
                frame_count = 0
            if bbox:
                x1, y1, x2, y2 = bbox
                x1, x2 = max(0, x1), min(frame_bgr.shape[1], x2)
                y1, y2 = max(0, y1), min(frame_bgr.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    face_bgr = frame_bgr[y1:y2, x1:x2]
            result = face_roi_extractor.detect_landmarks(frame_bgr)
            ladmarked_frame = face_roi_extractor.draw_landmarks(frame_bgr, result)
            face_rois = face_roi_extractor.extract_face_roi(frame_bgr, result)
            if face_rois is not None:
                cv2.imshow("Face ROIs", face_rois)
            cv2.imshow("Webcam", frame)
            cv2.imshow("Landmarks", ladmarked_frame)

            key = cv2.waitKey(1) & 0xFF      # waits 1 ms, returns -1 if no key pressed
            if key == ord("q"):
                break

            frame_count += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_roi_extractor.landmarker.close()
