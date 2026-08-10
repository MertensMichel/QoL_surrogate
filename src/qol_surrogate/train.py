"""Training loop.
"""



import torch


def train(model, train_loader, val_loader, num_epochs=100, lr=0.001):
    """Train the model using the provided training and validation data loaders.

    Returns the trained model plus the full per-epoch loss history (one value
    per epoch each), so callers can plot loss vs. epoch afterward.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):

        model.train()  # Set the model to training mode
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(batch.x, batch.edge_index, batch.edge_weight)  # Forward pass
            loss = criterion(outputs, batch.y)  # Compute the loss
            train_loss += loss.item()  # Accumulate the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update the weights
        train_loss /= len(train_loader)  # Average training loss
        train_losses.append(train_loss)

        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # Disable gradient computation for validation
            for batch in val_loader:
                outputs = model(batch.x, batch.edge_index, batch.edge_weight)  # Forward pass
                loss = criterion(outputs, batch.y)
                val_loss += loss.item()
            val_loss /= len(val_loader)  # Average validation loss
        val_losses.append(val_loss)

        if epoch % 10 == 0:  # Print every 10 epochs
            print(f'Epoch [{epoch}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    return model, train_losses, val_losses

