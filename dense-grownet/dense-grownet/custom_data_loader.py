import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


class CustomDataLoader:
    def __init__(self, features, labels, batch_size=1, validation_size=0.0, raw_output=None):
        train_indices, val_indices, train_labels, val_labels = train_test_split(
            range(len(labels)), labels, test_size=0.2, stratify=labels, random_state=42
        )

        train_data = features[train_indices]
        val_data = features[val_indices]

        self.train_data_tensor = torch.tensor(train_data).float().to(device)
        self.train_labels_tensor = torch.tensor(train_labels).long().to(device)

        if raw_output is None:
            self.raw_output_train = None
            self.raw_output_val = None
        else:
            self.raw_output_train = raw_output[train_indices].clone().detach().float().to(device)
            self.raw_output_val = raw_output[val_indices].clone().detach().float().to(device)

        self.val_data_tensor = torch.tensor(val_data).float().to(device)
        self.val_labels_tensor = torch.tensor(val_labels).long().to(device)

        train_dataset = TensorDataset(self.train_data_tensor, self.train_labels_tensor)
        val_dataset = TensorDataset(self.val_data_tensor, self.val_labels_tensor)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    def get_train_loader(self):
        return self.train_loader

    def get_val_loader(self):
        return self.val_loader
