
# Breast Cancer Diagnostic Prediction

Welcome to the Medical Diagnosis project! This project aims to develop a machine learning model to assist in the diagnosis of breast cancer based on patient data and relevant features.


## Table Of Content

- [Dataset](#Dataset)
- [Features](#Features)
- [Acknowledgements](#Acknowledgements)
- [Installation](#Installation)
- [Usage](#Usage)
- [Model Training](#ModelTraining)
- [Evaluation](#Evaluation)
- [License](#License)
- [Contact](#Contact)
## Dataset

The dataset used for this project is sourced from Kaggle and contains detailed information about The breast cancer diagnostic used in this project contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The dataset is available [here](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data).

## Features

- Data-driven Diagnosis: Utilizes machine learning algorithms to predict breast cancer diagnosis based on patient data.
- Interactive Visualization: Visualizes model predictions and evaluation results through interactive plots for enhanced interpretation.
- Data Preprocessing: Includes preprocessing steps such as handling missing values, encoding categorical features, and scaling numerical features to ensure data quality and model performance.
  
## Acknowledgements

 - The dataset used in this project was sourced from [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data).
 - The project relies on the [scikit-learn](https://scikit-learn.org/) library for machine learning functionalities.


## Installation

To run this project locally, follow these steps:

1. Clone the repository:


```bash
git clone https://github.com/MathavanPandi/Machine-Learning-projects/tree/main/Breast%20Cancer%20Diagnostic%20Prediction

```

2. Install the required dependencies:

```bash
pip install -r requirements.txt

```

## Usage

```bash
python main.py

```

## Model Training

```bash
# Sample code snippet for model training
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the dataset
data = pd.read_csv('data.csv')

# Preprocess the data (handle missing values, encode categorical features, etc.)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# Initialize the Linear Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)


```

## Evaluation

```bash
# Sample code snippet for model evaluation
from sklearn.metrics import accuracy_score, classification_report
y_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, y_pred)
print("Training Accuracy:", train_acc)

y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print("Testing Accuracy:", test_acc)

# Make predictions on the test set
svm = SVC()
svm.fit(X_train,y_train)
prediction = svm.predict(X_test)
print(accuracy_score(y_test,prediction))
print(classification_report(y_test, prediction))
print(confusion_matrix(y_test,prediction))

# Model Interpretation (Coefficients)
coefficients = pd.DataFrame(model.coef_[0], index=X.columns, columns=['Coefficient'])
print("Model Coefficients:")
print(coefficients)

```

## License
This project is licensed under the MIT License.

## Contact

For any inquiries or suggestions, feel free to contact the project maintainer:

- Mathavan P
- Email: mathavan.inboxx@gmail.com
- LinkedIn: www.linkedin.com/in/mathavan07
