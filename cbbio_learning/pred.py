import pandas as pd
import numpy as np
import re

# Cargar el DataFrame desde el archivo CSV proporcionado
df = pd.read_csv('../data/merged_sf_metamorphic_regions.csv')

# Función para construir etiquetas basadas en los rangos
def build_labels(sequence, regions):
    labels = np.zeros(len(sequence), dtype=np.int64)  # Inicializar tensor de etiquetas con ceros
    region_re = r"(\d+)-(\d+)"  # Expresión regular para extraer los rangos "start-end"

    if isinstance(regions, float) and np.isnan(regions):
        return labels

    found_regions = re.findall(region_re, regions)

    for start, end in found_regions:
        labels[int(start) - 1:int(end)] = 1  # Marcar con 1 las posiciones dentro del rango

    return labels

# Crear una lista para almacenar las filas desdobladas
rows_list = []

# Para cada fila, crear una entrada para la cadena A y otra para la cadena B
for _, row in df.iterrows():
    # Verificar si las secuencias no son NaN antes de proceder
    if pd.notna(row['resSF.pdb.pairA.sequence']):
        # Crear la fila para la cadena A
        row_chainA = {
            'retainedID': row['retainedID_x'],
            'resSF.pdb.pair': row['resSF.pdb.pairA_x'],
            'resSF.pdb.pair.sequence': row['resSF.pdb.pairA.sequence'],
            'Unip.chain.pair': row['Unip.chain.pairA'],
            'pdb_id': row['pdb_id'],
            'chain': row['chain'],
            'source': 'A'
        }
        # Generar etiquetas para la cadena A
        row_chainA['labels'] = build_labels(row_chainA['resSF.pdb.pair.sequence'], row_chainA['resSF.pdb.pair'])
        rows_list.append(row_chainA)

    if pd.notna(row['resSF.pdb.pairB.sequence']):
        # Crear la fila para la cadena B
        row_chainB = {
            'retainedID': row['retainedID_x'],
            'resSF.pdb.pair': row['resSF.pdb.pairB_x'],
            'resSF.pdb.pair.sequence': row['resSF.pdb.pairB.sequence'],
            'Unip.chain.pair': row['Unip.chain.pairB'],
            'pdb_id': row['pdb_idB'],
            'chain': row['chainB'],
            'source': 'B'
        }
        # Generar etiquetas para la cadena B
        row_chainB['labels'] = build_labels(row_chainB['resSF.pdb.pair.sequence'], row_chainB['resSF.pdb.pair'])
        rows_list.append(row_chainB)

# Crear un nuevo DataFrame a partir de la lista de filas
df_separated = pd.DataFrame(rows_list)

# Mostrar las primeras filas del nuevo DataFrame separado con etiquetas
print(df_separated.head())


