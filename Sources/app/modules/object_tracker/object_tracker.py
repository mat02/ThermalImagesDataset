class ObjectTracker:
    def __init__(self, conf):
        if conf["tracker"] == "simple":
            from .simple_tracker import SimpleObjectTracker
            self.tracker = SimpleObjectTracker(conf["max_disappeared"], conf["max_distance"])
        else:
            raise RuntimeError(f"ObjectTracker not initialized. Unknown tracker {conf['tracker']}!")

    def track(self, image, detected_object_locations):
        return self.tracker.track(image, detected_object_locations)
