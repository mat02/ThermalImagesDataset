class FaceDetector:
    def __init__(self, conf):
        from .yolov8_face_detector import YoloV8FaceDetector

        self.detector = YoloV8FaceDetector(conf)

    def detect(self, image):
        return self.detector.detect(image)