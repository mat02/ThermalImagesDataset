import os

from modules.multi_image_capture import AltImageCapture
from dvgutils.pipeline.observable import observable


class AltCaptureImagePipe:
    def __init__(self, *args, **kwargs):
        self.image_capture = AltImageCapture(*args, **kwargs)
        self.stop = False
        observable.register("stop", self, self.on_stop)

    def __iter__(self):
        return self.generator()

    def on_stop(self):
        self.stop = True

    def generator(self):
        idx = 0
        path = self.image_capture.path
        while not self.stop:
            filename, image, alt_image = self.image_capture.read_multi()

            if image is not None and alt_image is not None:
                if path != filename:
                    # Get rid of input path from filename
                    filename = os.path.relpath(filename, start=path)
                else:
                    filename = os.path.basename(filename)
                # Getting the name of the file without the extension
                name = os.path.splitext(filename)[0]

                data = {
                    "idx": idx,
                    "path": path,
                    "filename": filename,
                    "name": name,
                    "image": image,
                    "alt_image": alt_image
                }
                idx += 1
                yield data
            else:
                break
