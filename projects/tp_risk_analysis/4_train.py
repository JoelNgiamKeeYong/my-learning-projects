###################################################################################################################################
# Model Training for Synthetic Transfer Pricing Dataset

# This script trains a machine learning model (Random Forest Classifier) on the preprocessed synthetic transfer pricing dataset.

# The script provides the following functionalities:
#   1. Load the preprocessed and split dataset (training and testing sets) from disk.
#   2. Set up preprocessing pipelines for numerical and categorical features.
#   3. Define and configure the Random Forest Classifier model.
#   4. Combine preprocessing and modeling steps into a single pipeline.
#   5. Train the model on the training dataset.
#   6. Save the trained model pipeline to disk for future use.

# The script is designed to be modular and user-friendly, with clear console output to indicate the progress of each step.

# Usage:
#   1. Run the script: `python 4_train.py`.
#   2. The script will train the model and save it to the `./models` directory.

# Import relevant packages
import joblib  # For saving the trained model pipeline
import time  # For measuring training time
from colorama import Fore  # For colored console output

from sklearn.ensemble import RandomForestClassifier  # Importing the RandomForestClassifier
from sklearn.pipeline import Pipeline  # To create a pipeline for combining preprocessing and modeling
from sklearn.compose import ColumnTransformer  # For applying different preprocessing steps to columns
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # For scaling numerical data and encoding categorical data

from helpers.console_helpers import print_colored

# =================================================================================================================================
# 1. Load the Split Data
# =================================================================================================================================
print_colored("Starting model training...\n", color=Fore.MAGENTA, emoji="🤖")  

# Load the previously split data (X_train, X_test, y_train, y_test) saved after preprocessing
try:
    print_colored("Loading the split data... 📂", color=Fore.YELLOW, emoji="")
    X_train, X_test, y_train, y_test = joblib.load('./data/split_data.pkl') 
    print_colored("Split data loaded successfully! ✅\n", color=Fore.GREEN, emoji="")
except Exception as e:
    print_colored(f"Error loading data: {e} ❌", color=Fore.RED, emoji="")
    exit(1)

# =================================================================================================================================
# 2. Set Up
# =================================================================================================================================
print_colored("Setting up pipelines and preprocessors... 🔧", color=Fore.YELLOW, emoji="") 

# Define which columns are numerical and which are categorical
num_features = ['Transaction_Amount', 'Market_Benchmark_Price', 'Deviation_Percentage', 'Transaction_to_Benchmark_Ratio']
cat_features = ['Company_A', 'Company_B', 'Product_Type', 'Currency']

# Set up the preprocessing pipeline for numerical features: Scaling using StandardScaler
# Pipeline to apply scaling to numerical features
num_transformer = Pipeline(steps=[('scaler', StandardScaler())])  

# Set up the preprocessing pipeline for categorical features: One-hot encoding using OneHotEncoder
# OneHotEncoder ignores unknown categories during transformation
cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])  

# Combine the numerical and categorical preprocessing steps using a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),  # Apply numerical transformation to 'num_features' columns
        ('cat', cat_transformer, cat_features)   # Apply categorical transformation to 'cat_features' columns
    ]
)  

# Define the model to use: Random Forest Classifier
# Instantiate RandomForestClassifier with a fixed random state for reproducibility
print_colored("Choosing model... 👓", color=Fore.CYAN, emoji="") 
model = RandomForestClassifier(random_state=42)  

# Combine the preprocessing steps and model into one pipeline
print_colored("Finalizing set up... 🧹", color=Fore.CYAN, emoji="") 
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), # First apply the preprocessing steps (both numerical and categorical)
    ('classifier', model) # Then apply the model (Random Forest)
])  

print_colored("Set up completed! ✅\n", color=Fore.GREEN, emoji="")

# =================================================================================================================================
# 3. Train the Model
# =================================================================================================================================
print_colored("Initiating training process... 🤖", color=Fore.YELLOW, emoji="") 

# Start the timer to measure training time
start_time = time.time()

# Train the model using the training data
# Fit the pipeline (preprocessing + classifier) to the training data
pipeline.fit(X_train, y_train)  

# Calculate the training duration
training_duration = time.time() - start_time

print_colored("Training process completed! ✅\n", color=Fore.GREEN, emoji="")

# =================================================================================================================================
# 4. Save the Trained Model
# =================================================================================================================================
print_colored("Saving model... 💾", color=Fore.YELLOW, emoji="") 

# Save the trained pipeline (model + preprocessing steps) to disk using joblib
# Save the trained model for future use (prediction, further evaluation)
joblib.dump(pipeline, './models/transaction_risk_model.pkl')  

print_colored("Model saved successfully ✅\n", color=Fore.YELLOW, emoji="") 

print_colored(f"Model building completed in {training_duration:.2f} seconds!\n", color=Fore.GREEN, emoji="✅")
