###################################################################################################################################
# Data Exploration for Synthetic Transfer Pricing Dataset

# This script performs exploratory data analysis (EDA) on the synthetic transfer pricing dataset.

# The script provides the following functionalities:
#   1. Load the dataset from a CSV file.
#   2. Display the first 5 rows of the dataset for a quick overview.
#   3. Show summary statistics for numerical columns (e.g., mean, std, min, max).
#   4. Display general information about the dataset (e.g., data types, non-null counts).
#   5. Count the number of unique values in each column.
#   6. Check for missing values in the dataset.
#   7. Identify and count duplicate rows.
#   8. Compute and display the correlation matrix for numerical columns.

# The script is designed to be interactive, allowing the user to choose which exploration tasks to perform via a command-line interface (CLI).
# The user can continue exploring the dataset or exit the program after each task.

# Usage:
#   1. Run the script: `python 2_explore.py`.
#   2. Follow the prompts to explore the dataset interactively.

# Import relevant packages
import pandas as pd
from colorama import Fore
from tabulate import tabulate
import logging
import os
import argparse 

from helpers.console_helpers import print_colored

# =================================================================================================================================
# 1. Set Up
# =================================================================================================================================

CSV_FILE_PATH = './data/synthetic_transfer_pricing_data.csv'

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Perform exploratory data analysis on the synthetic transfer pricing dataset.")
parser.add_argument("--file", type=str, default="./data/synthetic_transfer_pricing_data.csv", help="Path to the CSV file")
args = parser.parse_args()
CSV_FILE_PATH = args.file

# Load the dataset
def load_dataset(file_path):
    """
    Load the dataset from the specified file path.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        print_colored(f"Loading dataset from {file_path}... 📂", color=Fore.CYAN, emoji="📊")
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return None

# =================================================================================================================================
# 2. Exploratory Functions
# =================================================================================================================================

# Show the first 5 rows of the dataset
def show_first_rows(df):
    print_colored("\nFirst 5 rows of the dataset... 📋", color=Fore.CYAN, emoji="")
    print(tabulate(df.head(), headers='keys', tablefmt='fancy_grid', showindex=False))

# Show summary statistics of numerical columns
def show_summary_statistics(df):
    print_colored("\nSummary statistics of numerical columns... 📈", color=Fore.CYAN, emoji="")
    # Round the statistics to 2 decimal places
    summary_stats = df.describe().round(2)
    print(tabulate(summary_stats, headers='keys', tablefmt='fancy_grid', showindex=True))

# Display info about the dataset (e.g., data types, non-null counts)
def show_dataset_info(df):
    print_colored("\nInfo about the dataset... ℹ️", color=Fore.CYAN, emoji="")
    df.info()

# Display the count of unique values in each column
def show_unique_value_counts(df):
    print_colored("\nUnique value counts in each column... 🔢", color=Fore.CYAN, emoji="")
    print(tabulate(df.nunique().reset_index(), headers=['Column', 'Unique Count'], tablefmt='fancy_grid', showindex=False))

# Check for missing values in the dataset
def check_missing_values(df):
    print_colored("\nChecking for missing values... 🚨", color=Fore.RED, emoji="")
    print(tabulate(df.isnull().sum().reset_index(), headers=['Column', 'Missing Values'], tablefmt='fancy_grid', showindex=False))

# Check for duplicate rows in the dataset
def check_duplicate_rows(df):
    print_colored("\nChecking for duplicate rows... 🔍", color=Fore.YELLOW, emoji="")
    duplicate_rows = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_rows}")

# Display the correlation matrix for numeric columns
def display_correlation_matrix(df):
    print_colored("\nCorrelation Matrix... 🔄", color=Fore.CYAN, emoji="")
    numeric_df = df.select_dtypes(include=[float, int])
    print(tabulate(numeric_df.corr(), headers='keys', tablefmt='fancy_grid'))

# =================================================================================================================================
# 3. CLI Functionality
# =================================================================================================================================

# Main function to handle the CLI
def main():
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)

    # Load the dataset
    df = load_dataset(CSV_FILE_PATH)
    if df is None:
        logging.error("The dataset is empty or could not be loaded.")
        return

    # Main loop for CLI
    while True:
        # Display the menu
        print_colored("\nData Exploration Menu\n", color=Fore.MAGENTA, emoji="")
        print("1. Show first 5 rows of the dataset")
        print("2. Show summary statistics of numerical columns")
        print("3. Display info about the dataset")
        print("4. Display unique value counts in each column")
        print("5. Check for missing values")
        print("6. Check for duplicate rows")
        print("7. Display correlation matrix")
        print("8. Exit")

        # Get user input
        choice = input("\nEnter your choice (1-8): ")

        # Handle user choice
        if choice == "1":
            show_first_rows(df)
        elif choice == "2":
            show_summary_statistics(df)
        elif choice == "3":
            show_dataset_info(df)
        elif choice == "4":
            show_unique_value_counts(df)
        elif choice == "5":
            check_missing_values(df)
        elif choice == "6":
            check_duplicate_rows(df)
        elif choice == "7":
            display_correlation_matrix(df)
        elif choice == "8":
            print_colored("\nExiting Data Exploration. Goodbye! 👋\n", color=Fore.GREEN, emoji="")
            break
        else:
            print_colored("Invalid choice. Please try again.", color=Fore.RED, emoji="❌")

        # Ask the user if they want to continue
        if choice != "8":
            continue_exploration = input("\nDo you want to continue exploring? (Y/N): ").strip().upper()
            if continue_exploration != "Y":
                print_colored("\nExiting Data Exploration. Goodbye! 👋\n", color=Fore.GREEN, emoji="")
                break

if __name__ == "__main__":
    main()