import torch
import torch.types
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score


def training_loop(model: nn.Module,
                  train_dataset: Dataset,
                  test_dataset: Dataset,
                  classification_threshold: float,
                  optimizer: Optimizer,
                  loss_f: nn.Module,
                  num_of_epochs: int,
                  batch_size: int):

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_of_epochs):
        
        model.train()   # Model training
        for i, data in enumerate(train_dataloader):
            images, labels = data

            optimizer.zero_grad()

            outputs = model(images)

            loss = loss_f(outputs, labels)
            loss.backward()

            optimizer.step()

        model.eval()    # Model validation
        with torch.no_grad():
            model.eval()
            labels = []
            outputs = []
            for i, data in enumerate(validate_dataloader):
                images, batch_labels = data
                
                labels += batch_labels.tolist()
                batch_outputs = model(images)[:, 1]
                batch_outputs = batch_outputs > classification_threshold
                outputs += batch_outputs.type(torch.int32).tolist()
                
            precision = precision_score(labels, outputs)
            recall = recall_score(labels, outputs)
            f1 = f1_score(labels, outputs)
            print(f'Epoch: {epoch}, Precisson: {precision}, Recall: {recall}, f1: {f1}')
