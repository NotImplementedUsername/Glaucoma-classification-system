import torch
from model import GlaucomaClassifier
from torch.nn import Module


def save_model(model: Module,  file_path: str) -> None:
    '''
        The function is used to save the model
    '''
    torch.save(model.state_dict(), file_path)

def load_model(file_path: str) -> GlaucomaClassifier:
    '''
        The function is used to load the model
    '''
    model = GlaucomaClassifier()
    return model.load_state_dict(torch.load(file_path, weights_only=True, map_location=torch.device('cpu')))
