import deeplake
from torch.utils.data import random_split


def load_data():
    # Load the full dataset
    ds = deeplake.load("hub://activeloop/objectron_book_train")

    # Determine the size of each split
    dataset_size = len(ds)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Create the train/val/test splits
    train_dataset, val_dataset, test_dataset = random_split(ds, [train_size, val_size, test_size])
    
    return train_dataset, val_dataset, test_dataset