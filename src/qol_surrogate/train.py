"""Training loop.
"""



import copy

import torch


def train(model, train_loader, val_loader, num_epochs=100, lr=0.001, patience=None, min_delta=0.0):
    """Train the model using the provided training and validation data loaders.

    If patience is set, stops early once validation loss hasn't improved
    (by more than min_delta) for `patience` consecutive epochs, and restores
    the best-seen weights before returning - the loop stopping early doesn't
    mean the *last* epoch's weights are the best ones, so they aren't just
    left in place.

    Returns the trained model plus the full per-epoch loss history (one value
    per epoch each - shorter than num_epochs if stopped early), so callers can
    plot loss vs. epoch afterward.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = torch.nn.MSELoss()

    train_losses, val_losses = [], []

    best_val_loss = float('inf')
    best_state_dict = None
    epochs_without_improvement = 0

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

        scheduler.step(val_loss)  # Adjust learning rate based on validation loss
        # Early stopping logic
        if patience is not None:
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f'Early stopping at epoch {epoch} (no improvement in {patience} epochs, '
                      f'best validation loss: {best_val_loss:.4f})')
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, train_losses, val_losses

