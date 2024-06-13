# Healthcare Heart Attack Prediction

## Project Overview

Cardiovascular diseases are the leading cause of death globally. This project aims to predict the risk of heart attacks using machine learning techniques on a dataset containing various health metrics and demographic information. The goal is to identify the causes and develop a system to effectively predict heart attacks.

## Dataset Description

The dataset includes the following variables:

- **Age**: Age in years
- **Sex**: 1 = male; 0 = female
- **cp**: Chest pain type
- **trestbps**: Resting blood pressure (in mm Hg on admission to the hospital)
- **chol**: Serum cholesterol in mg/dl
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- **restecg**: Resting electrocardiographic results
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (1 = yes; 0 = no)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: Slope of the peak exercise ST segment
- **ca**: Number of major vessels (0-3) colored by fluoroscopy
- **thal**: 3 = normal; 6 = fixed defect; 7 = reversible defect
- **Target**: 1 = presence of heart disease; 0 = absence of heart disease

## Project Tasks

1. **Preliminary Analysis**:
   - Perform preliminary data inspection.
   - Report the findings on the structure of the data, missing values, duplicates, etc.
   - Remove duplicates and treat missing values using an appropriate strategy.

2. **Data Exploration**:
   - Get a preliminary statistical summary of the data.
   - Identify and explore categorical variables.
   - Analyze the distribution of cardiovascular disease (CVD) across various factors such as age, sex, resting blood pressure, cholesterol levels, etc.
   - Use visualization tools like count plots, pair plots, etc.

3. **Model Building**:
   - Build a baseline model to predict the risk of a heart attack using logistic regression and random forest.
   - Perform feature selection using correlation analysis and logistic regression (leveraging standard error and p-values from statsmodels).

## Getting Started

To get started with this project, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies listed in `requirements.txt`.
3. Download the dataset `CEP 1_ Dataset.xlsx` and place it in the appropriate directory.
4. Run the Jupyter notebooks to explore the data and build models.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgements

We would like to thank the US Census Bureau for providing the California Census Data used in this project
