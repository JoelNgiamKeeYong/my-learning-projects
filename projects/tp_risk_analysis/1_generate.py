###################################################################################################################################
# Generate Synthetic Transfer Pricing Data

# This script generates a synthetic dataset for transfer pricing analysis.
# This script simulates transactions between companies, including the following features:
#   - Transaction_ID: Unique identifier for each transaction.
#   - Company_A: The first company involved in the transaction.
#   - Company_B: The second company involved in the transaction.
#   - Transaction_Amount: The monetary value of the transaction.
#   - Product_Type: The type of product involved (e.g., Raw Materials, Finished Goods, Services).
#   - Transaction_Date: The date when the transaction occurred.
#   - Currency: The currency used for the transaction (e.g., USD, EUR, JPY).
#   - Market_Benchmark_Price: The market benchmark price for the product.
#   - Deviation_Percentage: The percentage deviation of the transaction amount from the market benchmark price.
#   - Risk_Flag: The risk classification of the transaction (e.g., High, Medium, Low).

# The dataset is saved to a CSV file for further analysis.

# Usage:
#   1. Run the script: 'python 1_generate.py'
#   2. (Optional) View the csv document generated

# Import relevant packages
import pandas as pd
import numpy as np
import subprocess
import os
from colorama import Fore

from helpers.console_helpers import print_colored

# =================================================================================================================================
# 1. Configuration
# =================================================================================================================================

# Configuration
CSV_FILE_PATH = "./data/synthetic_transfer_pricing_data.csv" # Location to save the generated data
CSV_FULL_FILE_PATH = os.path.join(CSV_FILE_PATH) # Relative path to the CSV file
N_SAMPLES = 2000 # Number of products
RANDOM_SEED = 42 # Seed for the random number generator (ensures reproducibility)
TRANSACTION_MINIMUM_AMOUNT = 1000 # Minimum amount is $1,000
TRANSACTION_MAXIMUM_AMOUNT = 500000 # Minimum amount is $500,000
CURRENCIES = ["USD", "EUR", "JPY", "GBP", "AUD"]
PRODUCT_CATEGORIES = ["Raw Materials", "Finished Goods", "Services"]
PRODUCT_CATEGORIES_DISTRIBUTION = [0.4, 0.4, 0.2] # Probabilities of categories being selected
VARIANCE = 0.05 # Small random noise, e.g. 5% variance
START_DATE = "2019-01-01"
END_DATE = "2023-12-31"
HIGH_RISK_THRESHOLD_PERCENT = 10  # i.e. Deviation > 10% is High risk
LOW_RISK_THRESHOLD_PERCENT = -10  # i.e. Deviation < -10% is Low risk
OPEN_FILE_AFTER_GENERATION = True

# Ensure the data directory exists
os.makedirs(os.path.dirname(CSV_FULL_FILE_PATH), exist_ok=True)

# Set a random seed for reproducibility of results
np.random.seed(RANDOM_SEED)

# =================================================================================================================================
# 2. Generate Core Features (Independent Variables)
# =================================================================================================================================

# Company Names (Company A and Company B)
# Generate company names representing two entities in each transaction
Company_A = []
Company_B = []
for _ in range(N_SAMPLES):
    a, b = np.random.choice(range(1, 51), 2, replace=False)  # Select 2 unique company IDs, avoid case whereby A and B are same
    Company_A.append(f"Company_{a}")
    Company_B.append(f"Company_{b}")

# Transaction Amount
Transaction_Amount = np.random.uniform(TRANSACTION_MINIMUM_AMOUNT, TRANSACTION_MAXIMUM_AMOUNT, N_SAMPLES)

# Market Benchmark Price
# Slightly deviating from Transaction_Amount to simulate transfer pricing data with varying benchmarks
Market_Benchmark_Price = Transaction_Amount * np.random.uniform(0.8, 1.2, N_SAMPLES)

# Simulate real-world variability
# Add random noise to introduce small fluctuations into the data
noise_factor = np.random.normal(0, VARIANCE, N_SAMPLES) 
Transaction_Amount += Transaction_Amount * noise_factor
Market_Benchmark_Price += Market_Benchmark_Price * noise_factor

# Deviation Percentage
# To show how much the transaction deviates from the market benchmark
Deviation_Percentage = 100 * (Transaction_Amount - Market_Benchmark_Price) / Market_Benchmark_Price

# Transaction Dates
# Generate random transaction dates between 2019-01-01 and 2023-12-31
start_date = pd.to_datetime(START_DATE)
end_date = pd.to_datetime(END_DATE)

# Calculate a range of random days between the start and end date for each sample
date_range = end_date - start_date
random_days = np.random.randint(0, date_range.days, N_SAMPLES)

# Add random time deltas to the start date to get random transaction dates
Transaction_Date = start_date + pd.to_timedelta(random_days, unit='D')

# Simulate product types with specific risk influence
Product_Type = np.random.choice( 
    PRODUCT_CATEGORIES,  
    size=N_SAMPLES, 
    p=PRODUCT_CATEGORIES_DISTRIBUTION
)

# Simulate different currencies for transactions
Currency = np.random.choice(
    CURRENCIES, 
    size=N_SAMPLES
)

# =================================================================================================================================
# 3. Generate Risk Flag (Target Variable)
# =================================================================================================================================

# Initialize Risk_Flag as "Medium" by default
Risk_Flag = np.array(["Medium"] * N_SAMPLES)

# Apply generic risk rules for all product categories
Risk_Flag = np.where(
    Deviation_Percentage > HIGH_RISK_THRESHOLD_PERCENT, "High", Risk_Flag
)
Risk_Flag = np.where(
    Deviation_Percentage < LOW_RISK_THRESHOLD_PERCENT, "Low", Risk_Flag
)

# Introduce some variability by adding noise to Risk_Flag, randomly changing a small number of classifications
risk_variation = np.random.choice([0, 1, 2], size=N_SAMPLES, p=[0.95, 0.025, 0.025])  # 95% keep original, 5% random changes
Risk_Flag = np.where(risk_variation == 1, "High", Risk_Flag)
Risk_Flag = np.where(risk_variation == 2, "Low", Risk_Flag)

# =================================================================================================================================
# 4. Create and Format DataFrame
# =================================================================================================================================

# Create a DataFrame to hold all the generated features and data
df = pd.DataFrame({
    "Transaction_ID": range(1, N_SAMPLES + 1),  # Unique transaction identifier
    "Company_A": Company_A,  # Company A involved in the transaction
    "Company_B": Company_B,  # Company B involved in the transaction
    "Transaction_Amount": Transaction_Amount,  # Transaction value
    "Product_Type": Product_Type,  # Type of product involved in the transaction
    "Transaction_Date": Transaction_Date,  # Date when the transaction occurred
    "Currency": Currency,  # Currency used for the transaction
    "Market_Benchmark_Price": Market_Benchmark_Price,  # Market benchmark price for the product
    "Deviation_Percentage": Deviation_Percentage,  # Deviation percentage from the market benchmark
    "Risk_Flag": Risk_Flag  # Risk classification of the transaction
})

# Round the numerical columns for cleaner output (2 decimal places)
df['Transaction_Amount'] = df['Transaction_Amount'].round(2)
df['Market_Benchmark_Price'] = df['Market_Benchmark_Price'].round(2)
df['Deviation_Percentage'] = df['Deviation_Percentage'].round(2)

# =================================================================================================================================
# 5. Save Final Output
# =================================================================================================================================

# Save the generated dataset to a CSV file for further use
try:
    df.to_csv(CSV_FILE_PATH, index=False)
    print_colored("Data generation completed!", color=Fore.GREEN, emoji="✅")
    print_colored(f"Dataset saved to {CSV_FILE_PATH}", color=Fore.CYAN, emoji="📂")
except Exception as e:
    print_colored(f"Error saving dataset: {e}", color=Fore.RED, emoji="❌")

# Open the CSV file with the default application
if OPEN_FILE_AFTER_GENERATION:
    try:
        if os.name == "nt":  # Windows
            # Use the absolute path for os.startfile
            os.startfile(os.path.abspath(CSV_FULL_FILE_PATH))
        elif os.name == "posix":  # macOS or Linux
            # Use subprocess to open the file with the default application
            subprocess.run(["open", os.path.abspath(CSV_FILE_PATH)])  # macOS
            subprocess.run(["xdg-open", os.path.abspath(CSV_FILE_PATH)])  # Linux
        print_colored(f"File opened", color=Fore.GREEN, emoji="✅")
    except Exception as e:
        print_colored(f"Error opening file: {e}", color=Fore.RED, emoji="❌")