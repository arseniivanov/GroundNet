from torch.utils.data import random_split

import os
import requests

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
download_dataset('train')
download_dataset('test')



def load_data():
    # Load the full dataset

    # Determine the size of each split
    dataset_size = len(ds)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Create the train/val/test splits
    train_dataset, val_dataset, test_dataset = random_split(ds, [train_size, val_size, test_size])
    
    return train_dataset, val_dataset, test_dataset