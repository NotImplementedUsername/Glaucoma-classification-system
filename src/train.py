import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch.types
from torch.utils.data import DataLoader
from model import GlaucomaClassifier
from data import Eye_fundus_dataset
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

num_of_epochs = 10
batch_size = 20

def training_loop(num_of_epochs: int, batch_size: int):
    
    model = GlaucomaClassifier()
    classification_threshold = 0.5

    location_negative = Path('./data/negative')
    location_positive = Path('./data/positive')
    train_dataset = Eye_fundus_dataset()
    test_dataset = Eye_fundus_dataset()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    loss_f = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

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

    return model
