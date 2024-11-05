import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from model import GlaucomaClassifier
from data import Eye_fundus_dataset
from pathlib import Path


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

num_of_epochs = 10
batch_size = 20

def training_loop(num_of_epochs: int, batch_size: int):
    
    model = GlaucomaClassifier()

    location_negative = Path('./data/negative')
    location_positive = Path('./data/positive')
    train_dataset = Eye_fundus_dataset()
    test_dataset = Eye_fundus_dataset()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    loss_f = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch in range(num_of_epochs):
        
        for i, data in enumerate(train_dataloader):
            images, labels = data

            optimizer.zero_grad()

            outputs = model(images)

            loss = loss_f(outputs, labels)
            loss.backward()

            optimizer.step()

        with torch.no_grad:
            for i, data in enumerate(validate_dataloader):
                images, labels = data

                outputs = model(images)

                # TODO compute metrics
    
    return model