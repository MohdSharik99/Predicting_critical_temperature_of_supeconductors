markdown
Copy code
# Predicting Critical Temperature of Superconductors

This repository contains the code and data for predicting the critical temperature of superconductors. The project involves data preprocessing, feature selection, outlier removal, and building a predictive model using XGBoost.

## Project Structure

```plaintext
Predicting_Critical_Temperature_Of_Superconductors/
├── Data/
│   ├── train_df.csv
│   ├── test_df.csv
├── Functions.py
├── config.py
├── main.py
├── .gitignore
└── README.md
Requirements
Python 3.7 or higher
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
statsmodels
Setup
Clone the repository:

sh
Copy code
git clone https://github.com/MohdSharik99/Predicting_Critical_Temperature_Of_Superconductors.git
cd Predicting_Critical_Temperature_Of_Superconductors
Create and activate a virtual environment:

sh
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required packages:

sh
Copy code
pip install -r requirements.txt
Usage
Prepare the data:

Ensure the train_df.csv and test_df.csv files are placed in the Data/ directory.

Run the script:

sh
Copy code
python main.py
Configuration
Configuration settings for the project are in the config.py file. You can modify paths, model parameters, feature selection parameters, outlier removal parameters, and plot parameters as needed.

Project Details
Data Preprocessing
Reading Data: The training and testing data are read from CSV files located in the Data directory.
Outlier Removal: Outliers are removed using the Local Outlier Factor method.
Feature Selection: The best features are selected using mutual information regression.
Data Scaling: The selected features are scaled using StandardScaler.
Model Building
An XGBoost Regressor model is built and trained on the preprocessed data. The model's performance is evaluated using Mean Squared Error, Mean Absolute Error, and R-squared metrics.

Results
The results, including training and testing metrics, are printed to the console. A scatter plot showing the actual vs. predicted critical temperature is also generated.

Custom Functions
The custom functions for data preprocessing and model evaluation are defined in the Functions.py file.

.gitignore
The .gitignore file includes the following:

.ipynb_checkpoints/
__pycache__/

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
Special thanks to all the contributors and the open-source community for their invaluable support and resources.



This `README.md` provides an overview of the project, including setup instructions, usage, configuration details, and an explanation of the project structure. Adjust the details to better fit your specific project requirements as needed.




