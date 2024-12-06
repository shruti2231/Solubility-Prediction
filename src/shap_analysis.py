import pandas as pd
import joblib  # Use joblib for loading the model
import shap
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the dataset
data = pd.read_csv("data/combined_with_descriptors.csv")

# Select feature columns and target column
X = data[['MW', 'LogP', 'TPSA', 'HBA', 'HBD', 'RotB']]  # Features
y = data['Solubility']  # Target (log solubility)

# 2. Use a subset of the dataset for faster SHAP analysis
subset_size = 100  # Define the subset size
X_sample = X.sample(n=subset_size, random_state=42)

# 3. Load the trained model using joblib
model_path = 'models/best_solubility_model.pkl'  # Path to your saved model
model = joblib.load(model_path)

print("Model loaded successfully!")

# 4. Create SHAP explainer for Random Forest
# Using a smaller background sample for SHAP calculations
background = X.sample(n=50, random_state=42)  # Smaller background dataset
explainer = shap.TreeExplainer(model, data=background)

# 5. Calculate SHAP values for the sampled dataset
print("Starting SHAP calculations on the subset...")
shap_values = explainer.shap_values(X_sample)
print("SHAP calculations completed!")

# 6. Visualizations
# 6.1 Summary Plot (Overall feature importance)
shap.summary_plot(shap_values, X_sample)
plt.savefig('outputs/shap_summary_plot.png')  # Save the summary plot
print("Summary plot saved as 'shap_summary_plot.png'.")

# 6.2 Analyze Feature Importance using SHAP
# Calculate mean absolute SHAP values for each feature
shap_importance = pd.DataFrame({
    'Feature': X.columns,
    'SHAP Importance': np.abs(shap_values).mean(axis=0)
}).sort_values(by='SHAP Importance', ascending=False)

# Print feature importance
print("Feature Importance (Mean Absolute SHAP Values):")
print(shap_importance)

# 6.3 Save Feature Importance to CSV
shap_importance.to_csv('outputs/shap_feature_importance.csv', index=False)
print("Feature importance saved as 'shap_feature_importance.csv'.")
