from data_loader import load_model, run_predictions, BookAnnotationsDataset
import torch
from torch.utils.data import DataLoader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "model_epoch_14.pth"
    model = load_model(model_path, device)

    dataset = BookAnnotationsDataset("book_annotations_test")
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    run_predictions(model, data_loader, device)

if __name__ == '__main__':
    main()