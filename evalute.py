from architectures import MCUnetBackbone
from data_loader import load_data
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Check if GPU is available and set the device accordingly


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instantiate the model
    model = MCUnetBackbone()
    # Move the model to the chosen device
    model = model.to(device)    

    # Define a custom loss function
    def custom_loss(output, target):
        # Normalize the output and target vectors
        norm_output = torch.nn.functional.normalize(output, p=2, dim=1)
        norm_target = torch.nn.functional.normalize(target, p=2, dim=1)
        
        # Calculate the mean squared error between the normalized vectors
        loss = torch.mean((norm_output - norm_target) ** 2)
        return loss

    # Create an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Number of epochs
    num_epochs = 10
    # Batch size
    batch_size = 1

    train_dataset, val_dataset, test_dataset = load_data()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_loader_elements = len(train_loader.dataset)*100
    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        with tqdm(total=train_loader_elements, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in train_loader:
                # Move the inputs and targets to the device
                images = batch[0][0].to(device)
                camera_data = batch[1][0].to(device)
                plane_normals = batch[2][0].to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                predicted_normals = model(images, camera_data)

                # Compute loss
                loss = custom_loss(predicted_normals, plane_normals)

                # Backward pass
                loss.backward()

                # Optimize
                optimizer.step()

                # Update running loss
                running_loss += loss.item()
                pbar.update(images.shape[0])

            # Validation loop
            model.eval()  # Set the model to evaluation mode
            val_loss = 0.0
            with torch.no_grad():  # No gradients needed for validation
                for batch in val_loader:
                    images = batch[0][0].to(device)
                    camera_data = batch[1][0].to(device)
                    plane_normals = batch[2][0].to(device)

                    # Forward pass
                    predicted_normals = model(images, camera_data)

                    # Compute loss
                    loss = custom_loss(predicted_normals, plane_normals)

                    # Update validation loss
                    val_loss += loss.item()

                # Print average loss for the validation
                val_loss /= len(val_loader)
                print(f"Validation Loss: {val_loss}")

if __name__ == '__main__':
    main()