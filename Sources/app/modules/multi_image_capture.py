import os
import cv2

from dvgutils.fs import list_files
from dvgutils.modules.image_capture import Transform


class AltImageCapture:
    def __init__(self, conf):
        self.conf = conf
        self.path = conf["path"]
        self.valid_ext = conf["valid_ext"]
        self.alt_path = conf["alt_path"] if "alt_path" in conf else None
        self.alt_ext = conf["alt_ext"]
        self.contains = conf["contains"] if "contains" in conf else None
        self.level = conf["level"] if "level" in conf else None

        # Setup optional image transformation function (resizing, flipping, etc.)
        self.transform = Transform(conf["transform"]) if "transform" in conf else None
        self.alt_transform = Transform(conf["alt_transform"]) if "alt_transform" in conf else None

        if os.path.isfile(self.path):
            self.source = iter([self.path])
        else:
            self.source = list_files(self.path, self.valid_ext, self.contains, self.level)

    def read(self):
        try:
            filename = next(self.source)
            image = cv2.imread(filename)
            if self.transform:
                image = self.transform(image)
            return filename, image
        except StopIteration:
            return None, None
        
    def read_multi(self):
        try:
            filename = next(self.source)
            image = cv2.imread(filename)

            alt_filename = os.path.splitext(os.path.basename(filename))[0]
            if self.alt_path is not None:
                alt_path = os.path.dirname(filename).replace(self.path, self.alt_path)
            else:
                # Try one level up
                alt_path = os.path.join(os.path.dirname(filename), "..")
            alt_filename = os.path.join(alt_path, alt_filename + f".{self.alt_ext}")

            alt_image = cv2.imread(alt_filename, cv2.IMREAD_UNCHANGED)
            
            if self.transform:
                image = self.transform(image)
                alt_image = self.alt_transform(alt_image)

            return filename, image, alt_image
        except StopIteration:
            return None, None, None

