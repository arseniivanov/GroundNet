import cv2
from networkx import center
import numpy as np
import torch

from data_loader import get_frame_annotation
from utils import transform_points_to_world

from objectron.annotation_data_pb2 import Sequence

def project_normal_to_screen(normal, center_point, camera_data, image_shape):
    rotation_matrix = camera_data['rotation_matrix']
    projection_matrix = camera_data['projection_matrix']
    image_width, image_height = image_shape[1], image_shape[0]

    # Transform normal from world space to camera space
    camera_normal = np.dot(rotation_matrix, normal)
    
    #TODO Make center point 3d, needed for verification that normal/point is correctly transformec

    # Scale the normal for visibility and calculate its end point in world space
    normal_endpoint = center_point + camera_normal * 0.1

    # Project both center point and normal endpoint to screen space
    center_point_screen = project_point_to_screen(center_point, projection_matrix, image_width, image_height)
    normal_endpoint_screen = project_point_to_screen(normal_endpoint, projection_matrix, image_width, image_height)

    return center_point_screen, normal_endpoint_screen

def project_point_to_screen(point, projection_matrix, image_width, image_height):
    point_h = np.append(point, 1)
    projected_point = np.dot(projection_matrix, point_h)
    projected_point /= projected_point[3]

    x_pixel = (projected_point[1] + 1) * image_width / 2
    y_pixel = (1 - projected_point[0]) * image_height / 2

    import pdb;pdb.set_trace()
    return int(x_pixel), int(y_pixel)

def select_points_and_draw_normal(video_filename, camera_data_to_world, camera_data_to_screen):
    points = []

    cap = cv2.VideoCapture(video_filename)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_filename}")

    ret, image = cap.read()
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            points.append((x, y))
            cv2.imshow('image', image)

    cv2.imshow('image', image)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) != 3:
        print("Please select exactly 3 points.")
        return


    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x_tensor = torch.tensor(x_coords, dtype=torch.float32).unsqueeze(0)  # Shape [1, 3]
    y_tensor = torch.tensor(y_coords, dtype=torch.float32).unsqueeze(0)  # Shape [1, 3]

    point_list = (x_tensor, y_tensor)

    world_normal = transform_points_to_world(point_list, camera_data_to_world)
    world_normal = world_normal[0].cpu().numpy()
    world_normal = np.array([world_normal[0], -world_normal[1], -world_normal[2]]) # Flip y and z-coord

    # Project normal to screen and draw it
    center_point_screen, normal_endpoint_screen = project_normal_to_screen(
        world_normal, np.mean(points, axis=0), camera_data_to_screen, image.shape)
    cv2.line(image, center_point_screen, normal_endpoint_screen, (255, 0, 0), 2)
    cv2.imshow('image with normal', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

annotation_file = "book_annotations_train\\book_batch-1_4_annotation.pbdata"
video_filename = "book_annotations_train\\book_batch-1_4_video.MOV"
cam_data = get_frame_annotation(annotation_file)
cam_data = cam_data[0][0]
camera_data_batch = np.expand_dims(cam_data, axis=0)  # Now shape is (1, 20)
camera_data_tensor = torch.tensor(camera_data_batch, dtype=torch.float32)


camera_data_to_screen = {}

with open(annotation_file, 'rb') as pb:
    sequence = Sequence()
    sequence.ParseFromString(pb.read())

    for data in sequence.frame_annotations:
        #Extract necesseties for transform to screen-space
        proj = np.array(data.camera.projection_matrix).reshape(4, 4)
        camera_data_to_screen["projection_matrix"] = proj
        camera_data_to_screen["rotation_matrix"] = proj[:3, :3]

select_points_and_draw_normal(video_filename, camera_data_tensor, camera_data_to_screen)