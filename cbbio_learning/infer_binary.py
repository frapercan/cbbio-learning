import re
import pandas as pd
import torch
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5EncoderModel

from cbbio_learning.pred_binary import ProteinClassifier, ProteinDataset


# Función de inferencia
def make_inferences(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for sequences in dataloader:
            sequences = sequences[0].to(device)
            logits = model(sequences).squeeze(-1)
            preds = torch.sigmoid(logits) >= 0.5
            predictions.extend(preds.cpu().numpy())
    return predictions

# Función para procesar el encabezado y extraer la información relevante
def process_fasta_header(header):
    parts = header.split('|')
    protein_id = parts[1] if len(parts) > 1 else "Unknown"
    protein_name = parts[2] if len(parts) > 2 else "Unknown"
    description_parts = parts[2].split(' ') if len(parts) > 2 else []
    description = " ".join(description_parts[1:]) if len(description_parts) > 1 else "Unknown"
    organism_match = re.search(r'OS=([^ ]+)', header)
    organism = organism_match.group(1) if organism_match else None
    gene_match = re.search(r'GN=([^ ]+)', header)
    gene = gene_match.group(1) if gene_match else None
    taxon_match = re.search(r'OX=([^ ]+)', header)
    taxon = taxon_match.group(1) if taxon_match else None
    return protein_id, protein_name, description, organism, gene, taxon

# Cargar el modelo y los pesos guardados
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProteinClassifier().to(device)
model.load_state_dict(torch.load('results/protein_classifier_weights.pth', map_location=device))
model.eval()

# Leer el archivo FASTA y extraer secuencias junto con los encabezados
fasta_file = '/home/bioxaxi/PycharmProjects/cbbio-learning/data/doubt.fasta'
headers = []
sequences = []
protein_ids = []
protein_names = []
descriptions = []
organisms = []
genes = []
taxons = []

for record in SeqIO.parse(fasta_file, "fasta"):
    headers.append(record.id)
    sequences.append(str(record.seq))
    protein_id, protein_name, description, organism, gene, taxon = process_fasta_header(record.description)
    protein_ids.append(protein_id)
    protein_names.append(protein_name)
    descriptions.append(description)
    organisms.append(organism)
    genes.append(gene)
    taxons.append(taxon)

# Crear un DataFrame con toda la información extraída
df_proteome = pd.DataFrame({
    "header": headers,
    "seq": sequences,
    "protein_id": protein_ids,
    "protein_name": protein_names,
    "description": descriptions,
    "organism": organisms,
    "gene": genes,
    "taxon": taxons
})

# Reemplazar caracteres no válidos en las secuencias por "X"
df_proteome["seq"] = df_proteome["seq"].apply(lambda x: re.sub(r"[UZOB]", "X", x))
df_proteome['label'] = False

# Crear dataset y dataloader para inferencia
proteome_dataset = ProteinDataset(df_proteome, mode="seq", device=device)
proteome_dataloader = DataLoader(proteome_dataset, batch_size=1, shuffle=False)

# Realizar predicciones
predictions = make_inferences(model, proteome_dataloader, device)

# Añadir las predicciones al DataFrame
df_proteome['prediction'] = predictions

# Guardar el DataFrame con predicciones a un archivo CSV
df_proteome.to_csv('proteome_predictions.csv', index=False)

# Mostrar las primeras filas del DataFrame con predicciones
print(df_proteome.head())
