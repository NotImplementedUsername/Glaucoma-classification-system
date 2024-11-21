import torch
from model import GlaucomaClassifier
from data import Eye_fundus_dataset
from pathlib import Path
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from train import training_loop


def get_model():

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = GlaucomaClassifier()
    
    train_dataset = Eye_fundus_dataset(Path('./data/train/negative'), Path('./data/train/positive'))
    val_dataset = Eye_fundus_dataset(Path('./data/validate/negative'), Path('./data/validate/positive'))
    
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_function = CrossEntropyLoss()
    
    num_of_epochs = 100
    batch_size = 20

    classification_threshold = 0.5

    # TODO initialize model

    training_loop(model,
                  train_dataset, val_dataset,
                  classification_threshold,
                  optimizer, loss_function,
                  num_of_epochs, batch_size)



def initialize_model():
    pass


def save_model():
    pass

def load_model():
    pass
