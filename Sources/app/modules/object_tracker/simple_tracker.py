from .centroid_tracker import CentroidTracker
from copy import deepcopy


class SimpleObjectTracker:
    def __init__(self, max_disappeared=20, max_distance=80):
        # Initialize the frame dimensions (we'll set them as soon as we read the first frame from the video)
        self.w = None
        self.h = None

        # Instantiate our centroid tracker, then initialize a list to store each of our OpenCV correlation trackers,
        # followed by a dictionary to map each unique object ID to a TrackableObject
        self.centroid_tracker = CentroidTracker(max_disappeared, max_distance)
        self.object_tracks = {}

    def track(self, frame, object_locations):

        # If the frame dimensions are empty, set them
        if self.w is None or self.h is None:
            (self.h, self.w) = frame.shape[:2]

        # Use the centroid tracker to associate the
        # (1) old object centroids with
        # (2) the newly computed object centroids
        objects, bbox_dims, metadata, disappeared  = self.centroid_tracker.update(object_locations)
        current_objects = []

        # Loop over the tracked objects
        for (object_id, centroid) in objects.items():
            # Check to see if a trackable object exists for the current object ID
            to = self.object_tracks.get(object_id, None)

            # If there is no existing trackable object, create one
            if to is None:
                to = {
                    "object_id": object_id,
                    "centroids": [centroid],
                    "bbox_dims": bbox_dims[object_id],
                    "metadata_history": [metadata[object_id]],
                    "metadata": metadata[object_id],
                    "disappeared": disappeared[object_id],
                }
            # Otherwise, there is a trackable object so we can update it
            else:
                to["centroids"].append(centroid)
                to["metadata_history"].append(metadata[object_id])
                to["metadata"] = metadata[object_id]
                to["disappeared"] = disappeared[object_id]

            current_objects.append(to)

            # Store the trackable object in our dictionary
            self.object_tracks[object_id] = to

        return current_objects
