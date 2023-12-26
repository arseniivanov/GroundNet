from hmac import new
from re import I
from torch.utils.data import Dataset, random_split
from utils import transform_points_to_world, extract_points_from_heatmaps

import os
import requests
import numpy as np
import os
import requests
import cv2
import torch

from objectron.annotation_data_pb2 import Sequence
from architectures import MCUnetBackbone

def adjust_and_normalize_intrinsics(intrinsics, original_width, original_height, new_width, new_height):
    scale_x = new_width / original_width
    scale_y = new_height / original_height

    fx, fy, cx, cy = intrinsics[0], intrinsics[4], intrinsics[2], intrinsics[5]

    # Adjust for the new image size
    fx_adj = fx * scale_x
    fy_adj = fy * scale_y
    cx_adj = cx * scale_x
    cy_adj = cy * scale_y

    return [fx_adj, fy_adj, cx_adj, cy_adj]

def read_video(filename):
    """
    Read a video file and return its frames as a tensor.

    :param filename: Path to the video file.
    :return: A tensor containing the video frames.
    """

    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {filename}")
    
    frames = []
    f = 0

    new_width, new_height = 320, 240

    while True and f < 100: 
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame
        frame = cv2.resize(frame, (new_width, new_height))
        # Convert the frame from BGR (OpenCV default) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize pixel values to the range 0-1
        frame = frame / 255.0

        frames.append(frame)
        f+=1

    cap.release()

    frames_array = np.array(frames)

    # Convert the list of frames to a torch tensor
    # The shape of the tensor will be [num_frames, height, width, channels]
    # You might need to adjust the shape based on your model's input requirements
    frames_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2).float()

    return frames_tensor

def get_frame_annotation(annotation_filename):
    """Grab annotated frames from the sequence and return camera info and normals."""
    caminfo = []
    normals = []
    original_width, original_height = 1920, 1440
    new_width, new_height = 320, 240

    with open(annotation_filename, 'rb') as pb:
        sequence = Sequence()
        sequence.ParseFromString(pb.read())

        for data in sequence.frame_annotations:
            # Concatenate transform and intrinsics into a single 1D array for each frame

            transform = np.array(data.camera.transform)
            intrinsics = np.array(data.camera.intrinsics)
            rescaled_intrinsics = adjust_and_normalize_intrinsics(intrinsics, original_width, original_height, new_width, new_height)

            combined_info = np.concatenate([transform, rescaled_intrinsics])
            caminfo.append(combined_info)
            
            # Extract the normal vector
            normal = np.array(data.plane_normal)
            normals.append(normal)

    return np.array(caminfo), np.array(normals)

def download_dataset(dataset_type):
    """
    Download the specified dataset type ('train' or 'test').

    :param dataset_type: A string, either 'train' or 'test'.
    """
    public_url = "https://storage.googleapis.com/objectron"
    blob_path = public_url + f"/v1/index/book_annotations_{dataset_type}"
    video_ids = requests.get(blob_path).text.split('\n')

    # Create a directory for the dataset if it doesn't exist
    dataset_dir = f"book_annotations_{dataset_type}"
    os.makedirs(dataset_dir, exist_ok=True)

    # Download the data for each video
    for video_id in video_ids:
        if video_id:  # Check if the video_id is not empty
            # Replace slashes in video_id
            safe_video_id = video_id.replace('/', '_')

            video_filename = f"{public_url}/videos/{video_id}/video.MOV"
            metadata_filename = f"{public_url}/videos/{video_id}/geometry.pbdata"
            annotation_filename = f"{public_url}/annotations/{video_id}.pbdata"

            local_video_path = os.path.join(dataset_dir, f"{safe_video_id}_video.MOV")
            local_metadata_path = os.path.join(dataset_dir, f"{safe_video_id}_geometry.pbdata")
            local_annotation_path = os.path.join(dataset_dir, f"{safe_video_id}_annotation.pbdata")

            # Check if the files already exist before downloading
            if not os.path.exists(local_video_path):
                video = requests.get(video_filename)
                with open(local_video_path, "wb") as file:
                    file.write(video.content)

            if not os.path.exists(local_metadata_path):
                metadata = requests.get(metadata_filename)
                with open(local_metadata_path, "wb") as file:
                    file.write(metadata.content)

            if not os.path.exists(local_annotation_path):
                annotation = requests.get(annotation_filename)
                with open(local_annotation_path, "wb") as file:
                    file.write(annotation.content)

# Download train and test datasets
#download_dataset('train')
#download_dataset('test')

class BookAnnotationsDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.data_files = self._get_data_files()

    def _get_data_files(self):
        # Retrieve the list of video files in the dataset directory
        files = [f for f in os.listdir(self.dataset_dir) if f.endswith('_video.MOV')][:100]
        print(len(files))
        return files

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        base_filename = self.data_files[idx].rsplit('_', 1)[0]

        video_path = os.path.join(self.dataset_dir, f"{base_filename}_video.MOV")
        annotation_path = os.path.join(self.dataset_dir, f"{base_filename}_annotation.pbdata")

        # Load and process video
        video_tensor = read_video(video_path)

        # Parse annotation data
        caminfo, normals = get_frame_annotation(annotation_path)

        caminfo = caminfo[:100]
        normals = normals[:100]

        if len(normals[2]) != 3:
            print(video_path)
            exit()

        # Check for length mismatch between video frames and annotation data
        if video_tensor.shape[0] != len(normals):
            raise Exception(f"Frame length = {video_tensor.shape[0]} does not match normal length = {len(normals)} for entry {video_path}")

        # Convert caminfo and normals to tensors
        caminfo_tensor = torch.tensor(caminfo, dtype=torch.float32)
        normals_tensor = torch.tensor(normals, dtype=torch.float32)

        return video_tensor, caminfo_tensor, normals_tensor

def load_data(validation_split=0.2):
    # Load the entire dataset
    full_dataset = BookAnnotationsDataset('book_annotations_train')

    # Calculate the size of each split
    total_size = len(full_dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size

    # Randomly split the dataset into training and validation
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Load the test dataset
    test_dataset = BookAnnotationsDataset('book_annotations_test')

    return train_dataset, val_dataset, test_dataset

def load_model(model_path, device):
    model = MCUnetBackbone().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def run_predictions(model, data_loader, device):
    with torch.no_grad():
        for batch in data_loader:
            images, camera_data, plane_normals = batch
            images, camera_data, plane_normals = images[0].to(device), camera_data[0].to(device), plane_normals[0].to(device)

            # Predict
            heatmaps = model(images)

            point_predictions = extract_points_from_heatmaps(heatmaps)
            predicted_normals = transform_points_to_world(point_predictions, camera_data)
            
            # Compare and print
            print("Real Normals:", plane_normals.cpu().numpy())
            print("Predicted Normals:", predicted_normals.cpu().numpy())

def visualize_frame_annotation(annotation_filename, video_filename, resize = True):
    new_width, new_height = 320, 240

    # Load video file
    cap = cv2.VideoCapture(video_filename)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_filename}")

    with open(annotation_filename, 'rb') as pb:
        sequence = Sequence()
        sequence.ParseFromString(pb.read())

        frame_idx = 0
        for data in sequence.frame_annotations:
            ret, frame = cap.read()
            orignial_width, original_height, _ = frame.shape
            if not ret:
                break

            # Resize and convert the frame for visualization
            if resize:
                frame = cv2.resize(frame, (new_width, new_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Draw annotations on the frame
            frame_with_annotations = draw_annotations_on_frame(frame, data)

            # Display the frame with annotations
            cv2.imshow('Frame with Annotations', frame_with_annotations)
            cv2.waitKey(0)  # Press any key to proceed to the next frame

            frame_idx += 1

    cap.release()

def draw_annotations_on_frame(frame, annotation_data):
    # Extract the normal vector
    world_normal = np.array(annotation_data.plane_normal)
    world_normal = np.array([world_normal[0], -world_normal[1], -world_normal[2]]) # Flip y and z-coord
    world_normal /= np.linalg.norm(world_normal)  # Normalize to unit length

    # Extract camera extrinsics (rotation and translation)
    proj = np.array(annotation_data.camera.projection_matrix).reshape(4, 4)
    image_width, image_height = frame.shape[1], frame.shape[0]

    # Transform normal from world space to camera space
    # Assuming the normal requires only the rotational part of the transform
    rotation_matrix = proj[:3, :3]  # Extract rotation part of projection matrix
    camera_normal = np.dot(rotation_matrix, world_normal)

    # Draw keypoints if available
    if hasattr(annotation_data.annotations[0], 'keypoints'):
        for keypoint in annotation_data.annotations[0].keypoints:
            x, y = keypoint.point_2d.x, keypoint.point_2d.y
            cv2.circle(frame, (int(x*image_width), int(y*image_height)), 5, (0, 255, 0), -1)  # Green point

    #Project and draw center point below
    plane_center = np.array(annotation_data.plane_center)
    plane_center = np.array([plane_center[0], -plane_center[1], -plane_center[2]]) #Flip y and z-coord as projection is made in different format
    plane_c_h = np.append(plane_center, 1)

    # Apply the projection matrix
    h = np.dot(proj, plane_c_h)
    h /= h[3]

    # Convert these to pixel coordinates
    x_pixel = (h[1] + 1) * image_width / 2  # Use y as x
    y_pixel = (1 - h[0]) * image_height / 2  # use x as y

    cv2.circle(frame, (int(x_pixel), int(y_pixel)), 5, (0,0,255), -1)

    normal_endpoint = plane_center + camera_normal * 0.1  # Scale the normal for visibility
    normal_endpoint_h = np.append(normal_endpoint, 1)
    normal_endpoint_image = np.dot(proj, normal_endpoint_h)
    normal_endpoint_image /= normal_endpoint_image[3]
    end_x_pixel = (normal_endpoint_image[1] + 1) * image_width / 2
    end_y_pixel = (1 - normal_endpoint_image[0]) * image_height / 2

    # Draw the line
    cv2.line(frame, (int(x_pixel), int(y_pixel)), (int(end_x_pixel), int(end_y_pixel)), (255, 0, 0), 2)

    return frame
