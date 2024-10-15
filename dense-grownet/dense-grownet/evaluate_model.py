import torch
import torch.nn as nn

from .calculate_metrics import calculate_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, custom_train_loader, criterion, optimizer, num_epochs, num_features, batch_size, patience=None):
    best_val_loss, early_val_accuracy, early_train_loss = float("inf"), 0.0, float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i in range(0, custom_train_loader.train_data_tensor.size(0), batch_size):
            inputs, labels = custom_train_loader.train_data_tensor[i : i + batch_size], custom_train_loader.train_labels_tensor[i : i + batch_size]
            optimizer.zero_grad()

            outputs = None
            if custom_train_loader.raw_output_train is None:
                outputs = model(inputs.view(-1, num_features))
            else:
                outputs = model(inputs.view(-1, num_features), custom_train_loader.raw_output_train[i : i + batch_size])

            loss = criterion(outputs, labels, model)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

        model.eval()
        running_val_loss = 0.0

        unregularized_criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for i in range(0, custom_train_loader.val_data_tensor.size(0), batch_size):
                inputs, labels = custom_train_loader.val_data_tensor[i : i + batch_size], custom_train_loader.val_labels_tensor[i : i + batch_size]
                inputs, labels = inputs.to(device), labels.to(device)

                if custom_train_loader.raw_output_train is None:
                    outputs = model(inputs.view(-1, num_features))
                else:
                    outputs = model(inputs.view(-1, num_features), custom_train_loader.raw_output_val[i : i + batch_size])

                val_loss = unregularized_criterion(outputs, labels)
                running_val_loss += val_loss.item()

            avg_val_loss = running_val_loss / (len(custom_train_loader.get_val_loader()))

            val_accuracy, val_f1 = calculate_metrics(model, custom_train_loader)

            if (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch+1}, Loss: {running_loss / (len(custom_train_loader.get_train_loader()))}")
                print(f"Validation Loss: {avg_val_loss}")
                print(f"Val Accuracy: {val_accuracy:.4f}, Val F1-score: {val_f1:.4f}")
                print()

            if patience is None:
                continue
            elif avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_val_accuracy = val_accuracy
                early_train_loss = running_loss / len(custom_train_loader.get_train_loader())
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered. Best validation loss at epoch {best_epoch+1}: {best_val_loss:.4f}")
                print(f"Val Accuracy: {early_val_accuracy:.4f}")
                print(f"Early Train Loss: {early_train_loss}")
                break
