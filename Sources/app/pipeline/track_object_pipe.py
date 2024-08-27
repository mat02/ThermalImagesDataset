from modules.object_tracker import ObjectTracker


class TrackObjectPipe:

    def __init__(self, conf):
        self.tracker = ObjectTracker(conf)

    def __call__(self, data):
        return self.detect(data)

    def detect(self, data):
        image = data["image"]
        object_locations = data["object_locations"]

        # Track objects
        data["tracked_objects"] = { obj["object_id"]: obj for obj in self.tracker.track(image, object_locations) }
        data["tracked_faces"] = { id: obj["metadata"] for id, obj in data["tracked_objects"].items()}

        return data
