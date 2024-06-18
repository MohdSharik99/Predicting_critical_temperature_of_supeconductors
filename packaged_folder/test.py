# Importing important libraries
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Importing custom functions and configurations
from Functions import Evaluation_results
import config

warnings.filterwarnings('ignore')

def main():
    # Reading Test Data from Data Directory
    test_df = pd.read_csv(config.DATA_PATHS['test_data'])

    X_test = test_df.drop('critical_temp', axis=1)
    y_test = test_df['critical_temp']

    # Load the scaler and selected columns from train.py
    scaler = joblib.load('scaler.pkl')
    selected_columns = joblib.load('selected_columns.pkl')
    X_test_filtered = X_test[selected_columns]

    print(f'Number of testing points: ',len(X_test_filtered))
    # Scale the test data
    scaled_test_k_best = scaler.transform(X_test_filtered)

    # Load the trained model
    xgb_model = joblib.load('xgb_model.pkl')

    # Predict on the testing set
    xgb_test_preds = xgb_model.predict(scaled_test_k_best)
    
    # Testing results
    num_features = scaled_test_k_best.shape[1]
    test_metrics = Evaluation_results(y_test, xgb_test_preds, objective='test', num_features=num_features)
    print(test_metrics)

    # Plotting the results
    sns.set_style(config.PLOT_PARAMS['style'])
    sns.scatterplot(x=y_test, y=xgb_test_preds)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel(config.PLOT_PARAMS['xlabel'])
    plt.ylabel(config.PLOT_PARAMS['ylabel'])
    plt.title(config.PLOT_PARAMS['title'])
    plt.show()

if __name__ == "__main__":
    main()
