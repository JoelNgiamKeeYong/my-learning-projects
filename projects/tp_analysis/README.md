# Transfer Pricing Risk Assessment: Machine Learning Solution

==================================================================================================================================================

## Business Scenario

Transfer pricing refers to the prices at which transactions occur between subsidiaries or related entities within a multinational corporation. These transactions can include the sale of goods, provision of services, or transfer of intellectual property. For tax and financial reporting purposes, it's critical that transfer pricing adheres to `arm's length principles`, which means transactions should occur at market rates to prevent tax evasion and manipulation.

Multinational companies face challenges ensuring that transfer pricing complies with both local and international tax regulations. Mispricing can lead to costly audits, reputational damage, and financial penalties. Therefore, it's important to develop automated systems that can assess the `risk` of transfer pricing transactions in real-time.

==================================================================================================================================================

## Business Problem

Given the volume and complexity of transactions between multinational subsidiaries, identifying high-risk transfer pricing transactions manually can be tedious and prone to error. The business problem is to develop a `machine learning model` that can classify the risk of each transaction (high, medium, or low) based on factors such as:

- `Transaction Amount`
- `Market Benchmark Price`
- `Product Type`
- `Companies Involved`

The model should provide accurate predictions to assist audit teams and decision-makers by classifying each transaction's risk of violating transfer pricing rules. This can ensure that high-risk transactions are flagged for further review, minimizing compliance risks.

==================================================================================================================================================

## Solution Approach

The project follows a well-structured `end-to-end machine learning workflow` to automate transfer pricing risk classification. Below is a detailed breakdown of the steps:

### 1. **Data Generation**

- **Synthetic Data Creation**: A Python script generates synthetic transfer pricing data, simulating real-world transactions.
- **Features Included**:
  - `Transaction Amount`
  - `Market Benchmark Price`
  - `Product Type`
  - `Companies Involved`
  - `Deviation Percentage`: The difference between the transaction price and market price.
  - `Risk Flag`: The classification of the transaction risk (High, Medium, Low).
- **Purpose**: This step ensures the availability of a realistic dataset for model development and testing.

### 2. **Data Preprocessing**

- **Missing Data Handling**: Numerical and categorical columns are imputed with the median and most frequent values, respectively.
- **Feature Engineering**:
  - Created a new feature, `Transaction_to_Benchmark_Ratio`, to provide more meaningful insights into pricing deviations.
  - Extracted `Transaction_Year` and `Transaction_Month` from the `Transaction_Date` for temporal analysis.
- **Categorical Encoding**: Label encoding is applied to the `Risk_Flag` feature for machine learning compatibility.
- **Data Splitting**: The data is split into `training` (80%) and `testing` (20%) sets for model evaluation.

### 3. **Model Building**

- **Algorithm Selection**: A `Random Forest Classifier` is chosen for its robustness in handling both numerical and categorical data.
- **Preprocessing Pipeline**:
  - Numerical features are scaled using `StandardScaler`.
  - Categorical features are one-hot encoded using `OneHotEncoder`.
- **Training**: The model is trained on the preprocessed data and evaluated using `cross-validation` to ensure generalizability.

### 4. **Model Evaluation**

- **Metrics Used**:
  - `Accuracy`: The proportion of correct predictions.
  - `Confusion Matrix`: Visualizes the classification results (True Positives, False Positives, etc.).
  - `Classification Report`: Includes precision, recall, and F1-score for each class (High, Medium, Low risk).
  - `ROC-AUC Score`: Measures the model's ability to distinguish between classes.
  - `Log Loss`: Evaluates the confidence of the model's predicted probabilities.
- **Results**:
  - The model achieves high accuracy and ROC-AUC scores, demonstrating its effectiveness in classifying transaction risks.

### 5. **Model Deployment**

- **Saving the Model**: The trained model and preprocessing pipeline are saved using `joblib` for future use.
- **Integration**: The model can be integrated into a company's workflow to classify real-time transactions and flag high-risk cases for further review.

==================================================================================================================================================

## Outcome

The machine learning model provides an `automated risk classification system` to assist multinational corporations in:

- **Identifying High-Risk Transactions**: Early detection of transactions that may violate transfer pricing rules.
- **Improving Compliance**: Ensuring adherence to local and international tax regulations.
- **Streamlining Audits**: Reducing manual effort by flagging only high-risk transactions for review.
- **Reducing Errors**: Minimizing the risk of manual errors and bias in decision-making.

==================================================================================================================================================

## Key Skills Demonstrated

This project showcases the following **end-to-end machine learning skills**, which are critical for any data science or machine learning role:

### 1. **Problem Definition**

- Translated a business challenge (transfer pricing compliance) into a machine learning problem.
- Identified key features and target variables for risk classification.

### 2. **Data Generation and Preprocessing**

- Created synthetic data to simulate real-world transfer pricing transactions.
- Handled missing values, performed feature engineering, and encoded categorical variables.
- Split data into training and testing sets for model evaluation.

### 3. **Model Development**

- Built a `Random Forest Classifier` using `scikit-learn`.
- Structured the preprocessing and modeling steps into a `Pipeline` for scalability and reproducibility.
- Trained the model using cross-validation to ensure robustness.

### 4. **Model Evaluation**

- Evaluated the model using multiple metrics, including accuracy, precision, recall, F1-score, ROC-AUC, and log loss.
- Visualized results using confusion matrices and classification reports.

### 5. **Model Deployment**

- Saved the trained model and preprocessing pipeline using `joblib` for future use.
- Demonstrated the ability to integrate the model into a production environment for real-time risk assessment.

### 6. **Project Management**

- Organized the project into modular scripts for data generation, preprocessing, model training, evaluation, and deployment.
- Used version control (Git) to track changes and ensure reproducibility.

==================================================================================================================================================

## Technical Tools and Libraries

The following tools and libraries were used in this project:

- **Python**: Primary programming language.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computations.
- **Scikit-learn**: Machine learning algorithms and evaluation metrics.
- **Joblib**: Saving and loading models.
- **PrettyTable**: Displaying structured results in the console.
- **Colorama**: Adding colored console output for better readability.
- **Matplotlib/Seaborn**: Data visualization (optional for advanced visualizations).

==================================================================================================================================================

## Files and Directories

The project is organized into the following structure:

- **`data/`**: Contains synthetic and preprocessed transfer pricing data.
- **`models/`**: Stores the trained machine learning model.
- **Scripts**:
  - `1_generate.py`: Generates synthetic transfer pricing data.
  - `2_explore.py`: Performs exploratory data analysis (EDA) on the dataset.
  - `3_clean.py`: Cleans and preprocesses the data for modeling.
  - `4_train.py`: Trains the Random Forest Classifier and saves the model.
  - `5_evaluate.py`: Evaluates the model's performance on the test set.
  - `6_reset.py`: Resets the project by deleting files in the `data` and `models` folders.

==================================================================================================================================================

## Conclusion

This project demonstrates my ability to **manage an end-to-end machine learning project**, from problem understanding and data generation to model deployment. By leveraging synthetic data, robust preprocessing techniques, and a Random Forest Classifier, I developed a solution that automates transfer pricing risk assessment for multinational corporations. This project highlights my technical expertise in:

- **Data preprocessing and feature engineering**
- **Model development and evaluation**
- **Deployment and integration of machine learning models**

The skills and methodologies showcased in this project are directly applicable to real-world business challenges, making me a strong candidate for roles in data science and machine learning.

==================================================================================================================================================

## How to Run the Project

1. **Install Dependencies**:

   - pip install -r requirements.txt

2. **Run the Scripts (in this order)**:

   - python 1_generate.py
   - python 2_explore.py
   - python 3_clean.py
   - python 4_train.py
   - python 5_evaluate.py
   - python 6_reset.py
