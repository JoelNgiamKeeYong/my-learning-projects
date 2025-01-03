###################################################################################################################################
# Data Preprocessing for Synthetic Transfer Pricing Dataset

# This script performs data cleaning, preprocessing, and feature engineering on the synthetic transfer pricing dataset.

# The script provides the following functionalities:
#   1. Load the dataset from a CSV file.
#   2. Handle missing values in numerical and categorical columns using appropriate imputation strategies.
#   3. Check for and remove duplicate rows.
#   4. Perform feature engineering to create new features (e.g., transaction-to-benchmark ratio, transaction year, and month).
#   5. Encode categorical variables into numerical labels for machine learning compatibility.
#   6. Split the dataset into training and testing sets (80% train, 20% test).
#   7. Save the preprocessed data and split datasets to disk for further use.

# The script is designed to be modular and user-friendly, with clear console output to indicate the progress of each step.

# Usage:
#   1. Run the script: `python 3_clean.py`.
#   2. The script will preprocess the dataset and save the results to the `./data` directory.

# Import relevant packages
import pandas as pd  # Importing pandas to handle data manipulation
import joblib  # For saving models or data structures to disk
import os
from colorama import Fore  # For colored console output to make the script more interactive and user-friendly

from sklearn.impute import SimpleImputer  # To fill missing values in numerical and categorical columns
from sklearn.preprocessing import LabelEncoder  # For encoding categorical variables into numeric values
from sklearn.model_selection import train_test_split  # To split the dataset into training and testing sets

from helpers.console_helpers import print_colored

# =================================================================================================================================
# 1. Configure and Load Dataset
# =================================================================================================================================

CSV_FILE_PATH = './data/synthetic_transfer_pricing_data.csv'
TEST_SIZE = 0.2 # i.e. 0.2 --> 80% train, 20% test
RANDOM_SEED = 42

print_colored("Starting Data Preprocessing...\n", color=Fore.MAGENTA, emoji="🧹")  

print_colored("Loading the dataset... 📂", color=Fore.YELLOW, emoji="")  
try:
    # Load the dataset into a pandas DataFrame from the specified file path
    df = pd.read_csv(CSV_FILE_PATH)  
    print_colored("Dataset loaded successfully! ✅\n", color=Fore.GREEN, emoji="")  
except Exception as e:
    print_colored("Error loading the dataset... ❌", color=Fore.RED, emoji="")
    exit(1)

# =================================================================================================================================
# 2. Handle Missing Values
# =================================================================================================================================
print_colored("Handling missing values... 🚨", color=Fore.YELLOW, emoji="") 

# Define the columns that are numerical and categorical to apply appropriate imputations
numerical_columns = ['Transaction_Amount', 'Market_Benchmark_Price', 'Deviation_Percentage']
categorical_columns = ['Company_A', 'Company_B', 'Product_Type', 'Currency', 'Risk_Flag']

# Create imputers to handle missing values
num_imputer = SimpleImputer(strategy='median')  # Impute numerical missing values with the median
cat_imputer = SimpleImputer(strategy='most_frequent')  # Impute categorical missing values with the most frequent value

# Initiate process
print_colored("Imputing numerical missing values with the median... 🔢", color=Fore.CYAN,emoji="")
print_colored("Imputing categorical missing values with the most frequent value... 🔡", color=Fore.CYAN,emoji="")

# Apply the imputers to the respective columns
df[numerical_columns] = num_imputer.fit_transform(df[numerical_columns])
df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])

print_colored("Missing values handled successfully! ✅\n", color=Fore.GREEN, emoji="") 

# =================================================================================================================================
# 3. Check for Duplicate Rows
# =================================================================================================================================
print_colored("Checking for duplicate rows... 🧐", color=Fore.YELLOW, emoji="")

duplicate_rows = df.duplicated().sum()  # Check for duplicate rows in the DataFrame
if duplicate_rows > 0:
    # Print the number of duplicates found
    print_colored(f"Found {duplicate_rows} duplicate rows. Removing duplicates... 🧹", color=Fore.CYAN, emoji="") 
    # Remove duplicate rows
    df = df.drop_duplicates() 
    print_colored("Duplicate rows removed successfully! ✅\n", color=Fore.GREEN, emoji="") 
else:
    print_colored("No duplicate rows found! ✅\n", color=Fore.GREEN, emoji="") 

# =================================================================================================================================
# 4. Feature Engineering
# =================================================================================================================================
print_colored("Initiating feature engineering... 🔧", color=Fore.YELLOW, emoji="") 

# Creating new features based on existing data
# Ratio of transaction amount to benchmark price
print_colored("Creating feature: Ratio of transaction amount to benchmark price ... 🔢", color=Fore.CYAN,emoji="")
df['Transaction_to_Benchmark_Ratio'] = df['Transaction_Amount'] / df['Market_Benchmark_Price']  
# Convert 'Transaction_Date' to datetime format
print_colored("Converting 'Transaction Date' to datetime format ... 📅", color=Fore.CYAN,emoji="")
df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], dayfirst=False)  
# Extract year from the 'Transaction_Date' column
print_colored("Creating feature: Transaction Year ... 📅", color=Fore.CYAN,emoji="")
df['Transaction_Year'] = df['Transaction_Date'].dt.year  
# Extract month from the 'Transaction_Date' column
print_colored("Creating feature: Transaction Month ... 📅", color=Fore.CYAN,emoji="")
df['Transaction_Month'] = df['Transaction_Date'].dt.month  
# Drop the original 'Transaction_Date' column as we now have year and month
print_colored("Removing 'Transaction Date' feature ... 🗑️", color=Fore.CYAN,emoji="")
df = df.drop(columns=['Transaction_Date'])  

print_colored("Feature engineering completed! ✅\n", color=Fore.GREEN, emoji="")

# =================================================================================================================================
# 5. Encoding Categorical Variables
# =================================================================================================================================
print_colored("Encoding categorical variables... 🔠", color=Fore.YELLOW, emoji="")

# Initialize the LabelEncoder for encoding categorical variables into numerical labels
label_encoder = LabelEncoder()  

# Encode the 'Risk_Flag' column with integer labels (Low=0, Medium=1, High=2)
print_colored("Encoding 'Risk Flag' with integer values (Low=0, Medium=1, High=2) ... ➡️", color=Fore.CYAN,emoji="")
df['Risk_Flag'] = label_encoder.fit_transform(df['Risk_Flag'])  

print_colored("Categorical variables encoded! ✅\n", color=Fore.GREEN, emoji="") 

# =================================================================================================================================
# 6. Split the Data into Training and Test Sets
# =================================================================================================================================
print_colored("Splitting the data into training and testing sets... 📦", color=Fore.YELLOW, emoji="") 

# Define the feature matrix (X) and target vector (y)
# Drop the target variable and any identifier columns
X = df.drop(columns=['Risk_Flag', 'Transaction_ID'])  
# The target variable is 'Risk_Flag'
y = df['Risk_Flag']  

# Split the data into training and testing sets (80% train, 20% test)
print_colored(f"Splitting data into training ({(1-TEST_SIZE)*100}%) and test sets ({(TEST_SIZE)*100}%) ...📋", color=Fore.CYAN,emoji="")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

print_colored("Data split into training and testing sets! ✅\n", color=Fore.GREEN, emoji="")

# =================================================================================================================================
# 7. Save the Preprocessed Data
# =================================================================================================================================
print_colored("Saving the processed data... 💾", color=Fore.YELLOW, emoji="") 

# Ensure the data directory exists
os.makedirs('./data', exist_ok=True)

# Save the processed DataFrame to CSV
df.to_csv('./data/processed_data.csv', index=False)

# Save the split data (training and testing sets) into a .pkl file using joblib
joblib.dump((X_train, X_test, y_train, y_test), './data/split_data.pkl') 

print_colored("Processed data saved successfully! ✅\n", color=Fore.GREEN, emoji="") 

print_colored("Data preprocessing completed successfully!\n", color=Fore.GREEN, emoji="✅") 