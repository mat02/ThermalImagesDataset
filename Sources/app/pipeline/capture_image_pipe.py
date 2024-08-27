import os
from copy import deepcopy

from dvgutils.modules import ImageCapture
from dvgutils.pipeline import observable


class CaptureImagePipe:
    def __init__(self, *args, **kwargs):
        self.org_args = deepcopy(args)
        self.path_is_list = False
        self.path_idx = 0
        self.path = args[0]['path']
        
        if isinstance(self.path, list):
            self.image_capture = None
            self.path_is_list = True

            args[0]['path'] = self.path[self.path_idx]
            self.path_idx += 1
            self.image_capture = ImageCapture(*args, **kwargs)
        else:
            self.image_capture = ImageCapture(*args, **kwargs)
        self.stop = False
        observable.register("stop", self, self.on_stop)

    def __iter__(self):
        return self.generator()

    def on_stop(self):
        self.stop = True

    def generator(self):
        idx = 0
        while not self.stop:
            path = self.image_capture.path
            filename, image = self.image_capture.read()

            if image is not None:
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
                    "image": image
                }
                idx += 1
                yield data
            else:
                if self.path_is_list and self.path_idx < len(self.path):
                    args = deepcopy(self.org_args)
                    args[0]['path'] = self.path[self.path_idx]
                    print(f'Next file path: {args[0]["path"]}')
                    self.path_idx += 1
                    self.image_capture = ImageCapture(*args)
                else:
                    break
