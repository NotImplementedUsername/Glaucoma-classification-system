import torch
import torch.types
import torch.nn as nn
from torch.optim import Optimizer, Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from utils import save_model, load_model
from data import EyeFundusDataset
from model import GlaucomaClassifier
from pathlib import Path


def training_loop(model: nn.Module,
                  train_dataset: Dataset,
                  validate_dataset: Dataset,
                  classification_threshold: float,
                  optimizer: Optimizer,
                  loss_f: nn.Module,
                  num_of_epochs: int,
                  batch_size: int) -> None:

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = model.to(device)    # Loading model to CPU or GPU

    highest_f1 = -1.0       # Current highest value of f1
    
    for epoch in range(num_of_epochs):
        
        model.train()   # Model training
        for i, data in enumerate(train_dataloader): # Loading samples
            images, labels = data

            images = images.to(device)      # Loading samples to CPU or GPU
            labels = labels.to(device)

            optimizer.zero_grad()           # Reseting gradients

            outputs = model(images)         # Classifing samples

            loss = loss_f(outputs, labels)  # Calculating loss
            loss.backward()                 # Calculating gradints

            optimizer.step()                # Model improvement
 
        model.eval()    # Model validation
        with torch.no_grad():   # Turning off gradient
            all_labels = []     # Lists for all results and thier labels
            all_outputs = []
            for i, data in enumerate(validate_dataloader):
                images, labels = data
                
                images = images.to(device)      # Loading samples to CPU or GPU
                labels = labels.to(device)
                
                all_labels += labels.tolist()   # Adding batch labels to list of all labels
                outputs = model(images)[:, 1]   # Getting probability for positive classification  
                outputs = outputs > classification_threshold    # Appling classification threshold
                all_outputs += outputs.type(torch.int32).tolist()   # Coversion from bool to int and adding classification results to list of all results
                
            precision = precision_score(labels, outputs)    # Calculating metrics
            recall = recall_score(labels, outputs)
            f1 = f1_score(labels, outputs)   
            
            if f1 > highest_f1:         # If f1 value of current model is higher, save current model
                highest_f1 = f1         # Updating current highest value of f1
                save_model(model, './highest_f1.pt')    # Saving model with highest value of f1

            print(f'Epoch: {epoch}, Precison: {precision}, Recall: {recall}, f1: {f1}') # Displaying results


def get_model() -> GlaucomaClassifier:
    '''
        The function is used to train model with default parameteres
    '''
    model = GlaucomaClassifier()    # Model initialization
    
    # Loading datasets for training and validation
    train_dataset = EyeFundusDataset(Path('./data/train/negative'), Path('./data/train/positive'))
    val_dataset = EyeFundusDataset(Path('./data/validate/negative'), Path('./data/validate/positive'))
    
    classification_threshold = 0.5

    optimizer = Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    
    num_of_epochs = 1000
    batch_size = 20

    # Training model
    training_loop(model,
                  train_dataset, val_dataset,
                  classification_threshold,
                  optimizer, loss_function,
                  num_of_epochs, batch_size)

    try:
        model = load_model('./highest_f1.pt')   # Try to load model with highest f1 score
    except:
        pass    # If model with highest f1 score is not available, last model is returned

    return model
