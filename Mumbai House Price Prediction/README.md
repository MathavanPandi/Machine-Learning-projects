
# Mumbai House Price Prediction

This machine learning project aims to predict house prices in Mumbai using a dataset obtained from Kaggle. The project employs a Linear Regression model to analyze various features and make accurate predictions regarding property prices in different areas of Mumbai.


## Table Of Content

- [Dataset](#Dataset)
- [Features](#Features)
- [Acknowledgements](#Acknowledgements)
- [Installation](#Installation)
- [Usage](#Usage)
- [Model Training](#Model_Training)
- [Evaluation](#Evaluation)
- [License](#License)
- [Contact](#Contact)
## Dataset

The dataset used for this project is sourced from Kaggle and contains detailed information about houses in Mumbai, including various features such as area size, number of bedrooms, location, etc. The dataset is available [here](https://www.kaggle.com/datasets/dravidvaishnav/mumbai-house-prices).
## Features

- Linear Regression Model: Utilizes a linear regression algorithm for predicting house prices.
- Interactive Visualization: Visualizes prediction results through interactive plots.
- Data Preprocessing: Includes preprocessing steps such as handling missing values and encoding categorical features.
## Acknowledgements

 - The dataset used in this project was sourced from [Kaggle](https://www.kaggle.com/datasets/dravidvaishnav/mumbai-house-prices).
 - The project relies on the [scikit-learn](https://scikit-learn.org/) library for machine learning functionalities.


## Installation

To run this project locally, follow these steps:

1. Clone the repository:


```bash
git clone https://github.com/MathavanPandi/Machine-Learning/tree/84528ea95bd71ea6298072e9dd836b59e7b79ab2/Mumbai%20House%20Price%20Prediction
cd mumbai-house-price-prediction


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
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv('mumbai_house_prices.csv')

# Preprocess the data (handle missing values, encode categorical features, etc.)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)


```

## Evaluation

```bash
# Sample code snippet for model evaluation
from sklearn.metrics import mean_squared_error

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

```

## License
This project is licensed under the MIT License.

## Contact

For any inquiries or suggestions, feel free to contact the project maintainer:

- Mathavan P
- Email: mathavan.inboxx@gmail.com
- LinkedIn: www.linkedin.com/in/mathavan07