###################################################################################################################################
# Project Reset Script

# This script resets the project by deleting all files within the 'data' and 'models' folders while preserving the folders themselves.

# The script provides the following functionalities:
#   1. Deletes all files in the 'data' folder.
#   2. Deletes all files in the 'models' folder.
#   3. Provides clear feedback on the deletion process using colored console output.
#   4. Preserves the folder structure even after all files are deleted.

# The script is designed to be modular and user-friendly, with clear console output to indicate the progress of each step.

# Usage:
#   1. Run the script: `python 6_reset.py`.
#   2. The script will delete all files in the 'data' and 'models' folders and display the results in the console.

# Import relevant packages
import os  # For interacting with the operating system (file handling)
from colorama import Fore  # For colored console output

from helpers.console_helpers import print_colored  # Custom helper for colored console output

# =================================================================================================================================
# 1. Start Project Reset
# =================================================================================================================================
print_colored("Starting Project Reset...\n", color=Fore.MAGENTA, emoji="♻️")  

# Define the paths to the folders to delete files from
data_folder = "data"
model_folder = "models"

# =================================================================================================================================
# 2. Deleting Files in the 'data' Folder
# =================================================================================================================================
print_colored("Deleting files in the 'data' folder... 📂", color=Fore.YELLOW, emoji="")

# Check if the 'data' folder exists and is a directory
if os.path.exists(data_folder) and os.path.isdir(data_folder):
    # Iterate over the files in the 'data' folder
    for filename in os.listdir(data_folder):
        file_path = os.path.join(data_folder, filename)
        # Check if the current item is a file (not a subfolder)
        if os.path.isfile(file_path):
            os.remove(file_path)  # Delete the file
            print_colored(f"Deleted data file: {file_path}... 🗑️", color=Fore.RED, emoji="")
    
    # Check if the folder is empty after deletion
    if not os.listdir(data_folder):
        print_colored(f"All files in the {data_folder} folder have been deleted. ✅", color=Fore.GREEN, emoji="")
else:
    # Folder not found message
    print_colored(f"{data_folder} folder not found! ⚠️", color=Fore.YELLOW, emoji="")

print_colored("'data' folder cleanup completed! ✅\n", color=Fore.GREEN, emoji="")

# =================================================================================================================================
# 3. Deleting Files in the 'models' Folder
# =================================================================================================================================
print_colored("Deleting files in the 'models' folder... 🤖", color=Fore.YELLOW, emoji="")

# Check if the 'models' folder exists and is a directory
if os.path.exists(model_folder) and os.path.isdir(model_folder):
    # Iterate over the files in the 'models' folder
    for filename in os.listdir(model_folder):
        file_path = os.path.join(model_folder, filename)
        # Check if the current item is a file (not a subfolder)
        if os.path.isfile(file_path):
            os.remove(file_path)  # Delete the file
            print_colored(f"Deleted model file: {file_path}... 🗑️", color=Fore.RED, emoji="")
    
    # Check if the folder is empty after deletion
    if not os.listdir(model_folder):
        print_colored(f"All files in the {model_folder} folder have been deleted. ✅", color=Fore.GREEN, emoji="")
else:
    # Folder not found message
    print_colored(f"{model_folder} folder not found! ⚠️", color=Fore.YELLOW, emoji="")

print_colored("'models' folder cleanup completed! ✅\n", color=Fore.GREEN, emoji="")

# =================================================================================================================================
# 4. Completion Feedback
# =================================================================================================================================
print_colored("Project reset completed!\n", color=Fore.GREEN, emoji="🔄")