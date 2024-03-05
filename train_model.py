import torch
import torch.nn as nn
from torch.nn import Module
from sklearn.model_selection import train_test_split
import time
from typing import List
from torch.utils.data import TensorDataset, DataLoader
from model import CNNRBAModel
from data_loader import create_dataset, DataGenerator


def evaluate(dataloader: DataLoader, model: Module, loss_func):
    with torch.no_grad():
        loss_sum, good_sum, total = 0, 0, 0

        for x, y in dataloader:
            y_pred = model(DataGenerator.one_hot_encoding(x))
            loss = loss_func(y_pred, y)

            loss_sum += loss.item()
            good_sum += torch.sum(torch.round(y_pred) == y).item()
            total += len(y)

        avg_loss = loss_sum / total
        avg_acc = good_sum / total

    return avg_loss, avg_acc


def train_model(rbns_files: List[str], max_epochs: int = 2, max_runtime: int = 3600,
                show_performance: bool = False) -> Module:
    model = CNNRBAModel()
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    samples, labels = create_dataset(rbns_files, 1000000)
    x_train, x_validation, y_train, y_validation = train_test_split(samples, labels, test_size=0.3, random_state=42)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
    validate_loader = DataLoader(TensorDataset(x_validation, y_validation), batch_size=64)

    start_time = time.time()

    for epoch in range(0, max_epochs):
        if max_runtime < time.time() - start_time:
            break

        epoch_start_time = time.time()
        model.train()

        for x, y in train_loader:
            if max_runtime < time.time() - start_time:
                break

            optimizer.zero_grad()
            loss = loss_func(model(DataGenerator.one_hot_encoding(x)), y)
            loss.backward()
            optimizer.step()

        if show_performance:
            model.eval()

            train_loss, train_acc = evaluate(train_loader, model, loss_func)
            val_loss, val_acc = evaluate(validate_loader, model, loss_func)

            epoch_time = time.time() - epoch_start_time

            print(r"Model performance at epoch number {}".format(epoch + 1))
            print(r"Epoch: {}\n Train loss: {}\n, Train accuracy: {}".format(epoch + 1, train_loss, train_acc))
            print(r"Validation loss: {}\n Validation accuracy: {}\n, Epoch time: {}".format(val_loss, val_acc,
                                                                                            epoch_time))

    return model
