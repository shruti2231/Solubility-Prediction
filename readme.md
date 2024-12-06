<<<<<<< HEAD
## Solubilty prediction
## Shruti khare
=======
# Aqueous Solubility Prediction Using Machine Learning

This project predicts aqueous solubility of organic compounds using machine learning, specifically a **Random Forest Regressor** model. The model is trained on molecular descriptors, and **SHAP** (Shapley Additive Explanations) is used for interpretability.

## Project Overview

- **Objective**: Predict aqueous solubility using molecular descriptors and provide insights using SHAP for model interpretability.
- **Methodology**:
  - **Data**: The dataset consists of molecular descriptors and experimental solubility values from the **AqSolDB** dataset.
  - **Model**: We use **Random Forest Regressor** for solubility prediction.
  - **Interpretability**: SHAP is used to explain the predictions and identify key factors driving solubility.

## Installation

### Requirements

Make sure you have Python 3.x installed along with the following dependencies:
- pandas
- numpy
- scikit-learn
- shap
- matplotlib
- rdkit
- streamlit

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Project Structure

```
Project/
│
├── data/                        # Folder containing input datasets
│   ├── combined_dataset.csv     # Dataset used for model training
│
├── src/                         # Source code
│   ├── data_preparation.py      # Data cleaning and processing
│   ├── model_training.py        # Random Forest model training
│   ├── shap_analysis.py        # SHAP analysis and feature importance
│   ├── streamlit_app.py        # Streamlit application
│
├── models/                      # Trained models
│   ├── best_solubility_model.pkl
│
├── outputs/                     # Results and plots
│   ├── shap_summary_plot.png
│   ├── shap_dependence_plot_logp.png
│   ├── shap_feature_importance.csv
│
├── README.md
├── requirements.txt
├── main.py
```

## How to Run

1. **Data Preparation**:
   - Data is located in the `data/` folder
   - Run data preparation script:
     ```bash
     python src/data_preparation.py
     ```

2. **Model Training**:
   - Train the Random Forest model:
     ```bash
     python src/model_training.py
     ```
   - Hyperparameter tuning is performed using GridSearchCV

3. **SHAP Analysis**:
   - Generate SHAP plots and analyze feature importance:
     ```bash
     python src/shap_analysis.py
     ```

4. **Streamlit App**:
   - Launch the web interface:
     ```bash
     streamlit run src/streamlit prediciton.py
     ```
   - Input molecular descriptors or SMILES strings to get solubility predictions with model insights

## Results

- The model achieves high performance in predicting aqueous solubility with good accuracy (R², MAE, RMSE)
- SHAP analysis reveals key features influencing solubility predictions:
  - LogP
  - Molecular Weight
  - TPSA

## Future Work

- **Model Enhancements**:
  - Explore Gradient Boosting and Neural Networks
  - Incorporate fingerprint-based features
- **Dataset Expansion**:
  - Validate on larger datasets
  - Add experimental solubility data
- **App Improvements**:
  - Add batch predictions
  - Integrate external chemical databases


## Authors

- Bhupendra Singh Yaduvanshi MT23114
- Shruti Khare MT23238

## Acknowledgements

- AqSolDB dataset by the Autonomous Energy Materials Discovery (AMD) research group
- SHAP for machine learning model interpretability
- RDKit for cheminformatics and descriptor calculations
>>>>>>> 3c2f705 (added readme.md)
