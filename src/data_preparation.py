import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("data/combined_dataset.csv")

# Feature engineering: Calculate molecular descriptors using RDKit
def compute_descriptors(smiles):
    # Convert SMILES string to RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    descriptors = {
        'MW': Descriptors.MolWt(mol),  # Molecular weight
        'LogP': Descriptors.MolLogP(mol),  # LogP (hydrophobicity)
        'TPSA': Descriptors.TPSA(mol),  # Topological polar surface area
        'HBA': Descriptors.NumHAcceptors(mol),  # Number of hydrogen bond acceptors
        'HBD': Descriptors.NumHDonors(mol),  # Number of hydrogen bond donors
        'RotB': Descriptors.NumRotatableBonds(mol),  # Number of rotatable bonds
    }
    return descriptors

# Apply descriptor computation to SMILES column
data['descriptors'] = data['SMILES'].apply(compute_descriptors)

# Filter out any rows with invalid SMILES (if any)
data = data.dropna(subset=['descriptors'])

# Convert descriptors into a DataFrame
descriptors_df = pd.DataFrame(data['descriptors'].tolist())

# Combine the original data and the descriptors into a single DataFrame
data_combined = pd.concat([data.reset_index(drop=True), descriptors_df.reset_index(drop=True)], axis=1)

# Drop the original 'descriptors' column if you don't need it anymore
data_combined = data_combined.drop(columns=['descriptors'])

# If you want to save the combined DataFrame to a new CSV file
output_file_path = 'data/combined_with_descriptors.csv'  # Update as needed
data_combined.to_csv(output_file_path, index=False)
print(f"Combined data saved to {output_file_path}")

