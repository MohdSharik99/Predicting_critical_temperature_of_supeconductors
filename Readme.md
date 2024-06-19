This repository contains the code and data for predicting the critical temperature of superconductors. The project involves data preprocessing, feature selection, outlier removal, and building a predictive model using XGBoost.

## Table of Contents
* Project Structure
* Requirements
* Setup
* Usage
* Configuration
* Project Details
* Data Preprocessing
* Model Building
* Results
* Custom Functions
* License
* Acknowledgments

## Project Structure
```
Predicting_Critical_Temperature_Of_Superconductors/
├── Data/
│   ├── train_df.csv
│   └── test_df.csv
├── Readme.md
├── Exploratory Data Analysis.ipynb
├── Modeling_with_Select_k_best_features_30.ipynb
└── packaged_folder/
    ├── Data/
    ├── Functions.py
    ├── config.py
    ├── requirements.txt
    ├── scaler.pkl
    ├── selected_columns.pkl
    ├── test.py
    ├── train.py
    └── xgb_model.pkl
```
## Requirements
* Python 3.7 or higher
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* xgboost
* statsmodels
  
## Setup

Clone the repository:
```
# Clone repository
git clone https://github.com/MohdSharik99/Predicting_Critical_Temperature_Of_Superconductors.git

cd Predicting_Critical_Temperature_Of_Superconductors/packaged_folder
```

## Create and activate a virtual environment:

### On macOS and Linux:
```

python -m venv venv
source venv/bin/activate
```

On Windows:

```
python -m venv venv
venv\Scripts\activate
```

## Install the required packages:

```
pip install -r requirements.txt
```
## Usage

### Prepare the data:
Ensure the train_df.csv and test_df.csv files are placed in the Data/ directory.

Run the script:

```
python train.py
```

For testing data run the script:

```
python test.py
```

## Configuration
Configuration settings for the project are in the config.py file. You can modify paths, model parameters, feature selection parameters, outlier removal parameters, and plot parameters as needed.

## Project Details
* Data Preprocessing
* Reading Data: The training and testing data are read from CSV files located in the Data directory.
* Outlier Removal: Outliers are removed using the Local Outlier Factor method.
* Feature Selection: The best features are selected using mutual information regression.
* Data Scaling: The selected features are scaled using StandardScaler.

## Model Building
An XGBoost Regressor model is built and trained on the preprocessed data. The model's performance is evaluated using Mean Squared Error, Mean Absolute Error, and R-squared metrics.

## Results
The results, including training and testing metrics, are printed to the console. A scatter plot showing the actual vs. predicted critical temperature is also generated.

## Custom Functions
The custom functions for data preprocessing and model evaluation are defined in the Functions.py file.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
Thanks to all the contributors and the open-source community for their invaluable support and resources.
