import re

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.optim as optim
from transformers import T5Tokenizer, T5EncoderModel
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Verificación de CUDA
if not torch.cuda.is_available():
    raise Exception("CUDA no está disponible. Se requiere una GPU con CUDA.")
device = torch.device("cuda")

# Definición del clasificador binario
class ProteinClassifier(nn.Module):
    def __init__(self):
        super(ProteinClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        ).to(device)

    def forward(self, embeddings):
        logits = self.classifier(embeddings)
        return logits

# Definición del conjunto de datos
class ProteinDataset(Dataset):
    def __init__(self, dataframe, model_name='Rostlab/ProstT5', device='cpu', mode='seq'):
        self.dataframe = dataframe
        self.embeddings = []
        self.labels = []
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5EncoderModel.from_pretrained(model_name).to(device)
        model.eval()

        with torch.no_grad():
            for _, row in dataframe.iterrows():
                sequence = row[mode]
                label = row['label']
                sequence_processed = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
                inputs = tokenizer(sequence_processed, return_tensors="pt", padding=True, truncation=True, max_length=512, add_special_tokens=True).to(device)
                outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                self.embeddings.append(embeddings.cpu().numpy())
                self.labels.append(label)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx]), torch.tensor(self.labels[idx], dtype=torch.float)

# Función para configuración del modelo
def setup_model():
    model = ProteinClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    return model, optimizer, loss_fn

# Función de entrenamiento y validación
def train_and_validate(model, train_dataloader, test_dataloader, optimizer, loss_fn, num_epochs):
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    for epoch in range(num_epochs):
        # Entrenamiento y validación...
        pass  # Omitido para brevedad
    return train_losses, val_losses, train_accuracies, val_accuracies


if __name__ == '__main__':
    # Cargar el dataset y dividir
    df = pd.read_csv('../../protein-metamorphisms-is/data/sf_dataset.csv')
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['label'])
    train_dataset = ProteinDataset(df_train, device=device)
    test_dataset = ProteinDataset(df_test, device=device)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    # Entrenamiento del modelo
    model, optimizer, loss_fn = setup_model()
    num_epochs = 80
    train_losses, val_losses, train_accuracies, val_accuracies = train_and_validate(model, train_dataloader, test_dataloader, optimizer, loss_fn, num_epochs)

    torch.save(model.state_dict(), 'results/binary_predictor_weights.pth')