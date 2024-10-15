import torch
from sklearn.metrics import accuracy_score, f1_score


def calculate_metrics(model, loader):
    model.eval()

    with torch.no_grad():
        outputs = model(loader.val_data_tensor, loader.raw_output_val)

        _, preds = torch.max(outputs, 1)

        all_preds = preds.cpu().numpy()
        all_labels = loader.val_labels_tensor.cpu().numpy()

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    return accuracy, f1
