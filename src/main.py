import torch
from data import EyeFundusDataset
from utils import get_model, save_model
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

if __name__ == "__main__":
    model = get_model()

    test_dataset = EyeFundusDataset(Path('./data/test/negative'), Path('./data/test/positive'))
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=True)
    classification_threshold = 0.5

    model.eval()    # Model testing
    with torch.no_grad():
        labels = []
        outputs = []
        for i, data in enumerate(test_dataloader):
            images, batch_labels = data
            
            labels += batch_labels.tolist()
            batch_outputs = model(images)[:, 1]
            batch_outputs = batch_outputs > classification_threshold
            outputs += batch_outputs.type(torch.int32).tolist()
            
        precision = precision_score(labels, outputs)
        recall = recall_score(labels, outputs)
        f1 = f1_score(labels, outputs)
        print(f'Model testing, Precisson: {precision}, Recall: {recall}, f1: {f1}')

    save_model(model, './model.pt')
    