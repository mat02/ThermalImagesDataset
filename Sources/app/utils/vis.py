import cv2
import numpy as np

from dvgutils import colors
from dvgutils.vis import put_text, rectangle_overlay


def visualize_image_info(vis_image, filename):
    put_text(vis_image, filename, (0, 0), org_pos="tl",
             bg_color=colors.get("white").bgr(), bg_alpha=0.5)


def visualize_frame_info(vis_image, frame_num, fps):
    h, w = vis_image.shape[:2]
    # Visualize frame number
    put_text(vis_image, f"{frame_num}", (w, h), org_pos="br",
             bg_color=colors.get("white").bgr(), bg_alpha=0.5)
    # Visualize FPS
    put_text(vis_image, f"{fps:.2f} fps", (0, h), org_pos="bl",
             bg_color=colors.get("white").bgr(), bg_alpha=0.5)

def visualize_face_locations(vis_image, face_locations, scale=(1.0, 1.0)):
    for id, face_location in face_locations.items():
        (start_x, start_y, end_x, end_y) = (face_location["xyxy"] * np.array(scale * 2)).astype('int32')
        cv2.rectangle(vis_image, (start_x, start_y), (end_x, end_y), colors.get("blue").bgr(), 2)
        # rectangle_overlay(vis_image, (start_x, start_y), (end_x, end_y), colors.get("blue").bgr(), 0.3)

        for (x, y) in face_location["kpts_xy"]:
            cv2.circle(vis_image, (int(x * scale[0]), int(y * scale[1])), radius=int(1 * min(*scale)), color=(255, 0, 0), thickness=-1)

        # kpts = [cv2.KeyPoint(int(x * scale[0]), int(y * scale[1]), 1) for (x, y) in face_location["kpts_xy"]]
        # cv2.drawKeypoints(vis_image, kpts, vis_image)

def visualize_object_locations(vis_image, object_locations):
    if object_locations:
        for object_location in object_locations:
            (start_x, start_y, end_x, end_y, label, confidence) = object_location
            cv2.rectangle(vis_image, (start_x, start_y), (end_x, end_y), colors.get("green").bgr(), 2)
            rectangle_overlay(vis_image, (start_x, start_y), (end_x, end_y), colors.get("green").bgr(), 0.5)
            put_text(vis_image, f"{label} {confidence:.2f}", (start_x - 1, start_y - 1), org_pos="bl",
                     bg_color=colors.get("green").bgr())


def save_thermo_image(image, filename, cmap=None, scale=1.0, save=True):
    if scale != 1.0:
        image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
    if cmap:
        image = cv2.applyColorMap(image, cv2.COLORMAP_INFERNO)
    
    if save:
        cv2.imwrite(filename, image)
    
    return image
    
def draw_keypoints(image, keypoints, scale=1.0):
    for (x, y) in keypoints:
        cv2.circle(image, (int(x * scale), int(y * scale)), radius=int(1 * scale), color=(255, 0, 0), thickness=-1)
    return image
