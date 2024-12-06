import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import joblib
import os

# Title and Description
st.title("Molecular Solubility Predictor")
st.write("""
This app predicts the **aqueous solubility** of a molecule based on its SMILES (Simplified Molecular Input Line Entry System) string.
""")

# Load the trained model
@st.cache_resource
def load_model():
    # Construct the relative path to the model
    model_path = os.path.join(os.path.dirname(__file__), "../models/best_solubility_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)

# Load the model
model = load_model()
# Function to compute molecular descriptors
def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    descriptors = {
        'MW': Descriptors.MolWt(mol),  # Molecular weight
        'LogP': Descriptors.MolLogP(mol),  # LogP (hydrophobicity)
        'TPSA': Descriptors.TPSA(mol),  # Topological polar surface area
        'HBA': Descriptors.NumHAcceptors(mol),  # Hydrogen bond acceptors
        'HBD': Descriptors.NumHDonors(mol),  # Hydrogen bond donors
        'RotB': Descriptors.NumRotatableBonds(mol),  # Rotatable bonds
    }
    return descriptors

# Function to predict solubility
def predict_solubility(smiles, model):
    descriptors = compute_descriptors(smiles)
    if descriptors is None:
        return None, None, "Invalid SMILES"
    descriptors_df = pd.DataFrame([descriptors])
    predicted_log_solubility = model.predict(descriptors_df)[0]
    predicted_solubility = 10 ** predicted_log_solubility
    solubility_status = "Soluble" if predicted_solubility > 0.01 else "Insoluble"
    return predicted_solubility, predicted_log_solubility, solubility_status

# Input SMILES string
smiles_input = st.text_input("Enter SMILES string")  # Default is 1,2-dichlorobenzene

# Predict Button
if st.button("Predict Solubility"):
    predicted_solubility, predicted_log_solubility, solubility_status = predict_solubility(smiles_input, model)
    if solubility_status == "Invalid SMILES":
        st.error("Invalid SMILES string. Please check your input.")
    else:
        st.success(f"**Log-Solubility:** {predicted_log_solubility}")
        st.success(f"**Solubility (mol/L):** {predicted_solubility}")
        st.success(f"**Solubility Status:** {solubility_status}")
