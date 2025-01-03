###################################################################################################################################
# Model Evaluation for Synthetic Transfer Pricing Dataset

# This script evaluates the performance of a trained machine learning model (Random Forest Classifier) on the test dataset.

# The script provides the following functionalities:
#   1. Load the split dataset (X_test, y_test) and the trained model from disk.
#   2. Generate predictions on the test dataset.
#   3. Evaluate the model using accuracy, confusion matrix, classification report, ROC-AUC, and log loss.
#   4. Display results in a structured and visually appealing format using PrettyTable.
#   5. Highlight good and bad performance metrics using colored output.

# The script is designed to be modular and user-friendly, with clear console output to indicate the progress of each step.

# Usage:
#   1. Run the script: `python 5_evaluate.py`.
#   2. The script will evaluate the model and display the results in the console.

# Import relevant packages
import joblib  # For loading the trained model and split data
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,  roc_auc_score, log_loss # For model evaluation metrics
from colorama import Fore  # For colored console output
from prettytable import PrettyTable  # For structured and visually appealing output

from helpers.console_helpers import print_colored, highlight_value  # Custom helper for colored console output

# =================================================================================================================================
# 1. Load Data and Model
# =================================================================================================================================
print_colored("Starting Model Evaluation...\n", color=Fore.MAGENTA, emoji="🔍")  

# Load the split data (X_train, X_test, y_train, y_test)
try:
    print_colored("Loading the split data... 📂", color=Fore.YELLOW, emoji="")
    X_train, X_test, y_train, y_test = joblib.load('./data/split_data.pkl')
    print_colored("Split data loaded successfully! ✅\n", color=Fore.GREEN, emoji="")
except Exception as e:
    print_colored(f"Error loading split data: {e} ❌", color=Fore.RED, emoji="")
    exit(1)

# Load the saved model
try:
    print_colored("Loading the trained model... 🤖", color=Fore.YELLOW, emoji="")
    model = joblib.load('./models/transaction_risk_model.pkl')
    print_colored("Model loaded successfully! ✅\n", color=Fore.GREEN, emoji="")
except Exception as e:
    print_colored(f"Error loading model: {e} ❌", color=Fore.RED, emoji="")
    exit(1)

# =================================================================================================================================
# 2. Generate Predictions
# =================================================================================================================================
print_colored("Generating predictions... 🎯", color=Fore.YELLOW, emoji="")

# Generate predicted labels
y_pred = model.predict(X_test)

# Generate predicted probabilities for ROC-AUC and Log Loss
y_pred_proba = model.predict_proba(X_test)

print_colored("Predictions generated successfully! ✅\n", color=Fore.GREEN, emoji="")

# =================================================================================================================================
# 3. Evaluate Model Performance
# =================================================================================================================================
print_colored("Evaluating model performance... 📊", color=Fore.YELLOW, emoji="")

# 3.1. Accuracy
# High accuracy scores indicate that the model is performing well overall.
# However, accuracy alone can be misleading if the dataset is imbalanced (e.g., one class has significantly more samples than others). In this case, the dataset seems balanced, so accuracy is a reliable metric.
accuracy = accuracy_score(y_test, y_pred)

# Create a PrettyTable for accuracy
accuracy_table = PrettyTable()
accuracy_table.field_names = ["Metric", "Score"]
accuracy_table.add_row(["Accuracy", f"{accuracy * 100:.2f}%"])

# 3.2. Confusion Matrix
# The confusion matrix shows the actual vs. predicted classifications for each class
conf_matrix = confusion_matrix(y_test, y_pred)

# Create a PrettyTable for Confusion Matrix
conf_matrix_table = PrettyTable()
conf_matrix_table.field_names = ["", "Predicted High", "Predicted Medium", "Predicted Low"]

# Add rows for True Labels
conf_matrix_table.add_row(["True High", conf_matrix[0][0], conf_matrix[0][1], conf_matrix[0][2]])
conf_matrix_table.add_row(["True Medium", conf_matrix[1][0], conf_matrix[1][1], conf_matrix[1][2]])
conf_matrix_table.add_row(["True Low", conf_matrix[2][0], conf_matrix[2][1], conf_matrix[2][2]])

# 3.3. Classification Report
# The classification report provides precision, recall, and F1-score for each class, as well as macro and weighted averages.
report = classification_report(y_test, y_pred, output_dict=True)

# Create a PrettyTable for Classification Report
classification_report_table = PrettyTable()
classification_report_table.field_names = ["Class", "Precision", "Recall", "F1-Score", "Support"]

# Add rows for each class and the overall metrics
for class_name, metrics in report.items():
    if class_name != "accuracy":  # Skip accuracy as it's already printed
        # Use highlight_value to conditionally color both good and bad values
        precision = highlight_value(metrics['precision'])
        recall = highlight_value(metrics['recall'])
        f1_score = highlight_value(metrics['f1-score'])
        support = f"{metrics['support']}"  # Support is just a number, no need for color

        classification_report_table.add_row([class_name.capitalize(), precision, recall, f1_score, support])

# 3.4. ROC-AUC Score
# ROC-AUC is calculated using One-vs-Rest for multi-class classification
# The ROC-AUC score measures the model's ability to distinguish between classes. A score of 0.97 (close to 1) indicates excellent performance.
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

# Create a PrettyTable for ROC-AUC
roc_auc_table = PrettyTable()
roc_auc_table.field_names = ["Metric", "Score"]
roc_auc_table.add_row(["ROC-AUC", f"{roc_auc:.2f}"])

# 3.5. Log Loss
# Log Loss measures the confidence of the model's predicted probabilities. A lower log loss indicates better-calibrated probabilities.
logloss = log_loss(y_test, y_pred_proba)

# Create a PrettyTable for Log Loss
logloss_table = PrettyTable()
logloss_table.field_names = ["Metric", "Score"]
logloss_table.add_row(["Log Loss", f"{logloss:.2f}"])

# =================================================================================================================================
# 4. Print Results
# =================================================================================================================================
print("\nModel Evaluation:")
print(accuracy_table)
print("\nConfusion Matrix:")
print(conf_matrix_table)
print("\nClassification Report:")
print(classification_report_table)
print("\nROC-AUC Score:")
print(roc_auc_table)
print("\nLog Loss:")
print(logloss_table)

# Finish with colored feedback
print_colored("\nModel Evaluation Completed! 🎉\n", color=Fore.GREEN, emoji="")