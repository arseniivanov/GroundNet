from architectures import MCUnetBackbone
from data_loader import load_data
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

# Define the MCUnetBackbone class with all the required layers

# Instantiate the model
model = MCUnetBackbone()

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
batch_size = 32

train_dataset, val_dataset, test_dataset = load_data()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for batch in train_loader:
        # Get the inputs and targets from the batch
        images = batch['image']
        plane_normals = batch['plane_normal']

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass: Compute predicted normals by passing images to the model
        predicted_normals = model(images)

        # Compute loss
        loss = custom_loss(predicted_normals, plane_normals)

        # Backward pass: Compute gradient of the loss with respect to model parameters
        loss.backward()

        # Perform an optimization step (parameter update)
        optimizer.step()

        # Print statistics
        running_loss += loss.item()

    # Print average loss for the epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {running_loss / len(train_loader)}")

