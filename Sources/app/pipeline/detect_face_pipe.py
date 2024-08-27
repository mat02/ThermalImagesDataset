from modules.face_detector import FaceDetector
import cv2


class DetectFacePipe:

    def __init__(self, conf):
        self.detector = FaceDetector(conf)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def __call__(self, data):
        return self.detect(data)

    def detect(self, data):
        image = data["image"]

        # Detect faces
        # image[:, :, 0] = self.clahe.apply(image[:, :, 0])
        # image[:, :, 1] = image[:, :, 0]
        # image[:, :, 2] = image[:, :, 0]
        # image = cv2.blur(image, (3, 3))
        # ret2,th2 = cv2.threshold(image[:, :, 0],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # image[image < ret2] = 0
        data["image"] = image

        data["object_locations"] = self.detector.detect(image)


        return data
