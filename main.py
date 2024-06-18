# Importing important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
import warnings

# Importing custom functions and configurations
from Functions import Remove_outliers_with_lof, Select_k_best_features, Evaluation_results
import config

warnings.filterwarnings('ignore')

def main():
    # Reading Data from Data Directory
    train_df = pd.read_csv(config.DATA_PATHS['train_data'])
    test_df = pd.read_csv(config.DATA_PATHS['test_data'])

    X_train = train_df.drop('critical_temp', axis=1)
    y_train = train_df['critical_temp']

    X_test = test_df.drop('critical_temp', axis=1)
    y_test = test_df['critical_temp']

    # Removing outliers
    new_train_X, new_train_y = Remove_outliers_with_lof(X_train, y_train, contamination=config.OUTLIER_REMOVAL_PARAMS['contamination'])

    # Selecting best Features
    k = config.FEATURE_SELECTION_PARAMS['k']
    X_train_filtered, scores = Select_k_best_features(new_train_X, new_train_y, k=k, score_func=mutual_info_regression)

    # Scaling Data
    selected_columns = X_train_filtered.columns
    X_test_filtered = X_test[selected_columns]

    scaler = StandardScaler()
    scaled_train_k_best = scaler.fit_transform(X_train_filtered)
    scaled_test_k_best = scaler.transform(X_test_filtered)

    # Final Model
    xgb_model = xgb.XGBRegressor(**config.MODEL_PARAMS)

    xgb_model.fit(scaled_train_k_best, new_train_y)
    xgb_train_preds = xgb_model.predict(scaled_train_k_best)

    # Training results
    print('XGBoost Results')
    num_features = scaled_train_k_best.shape[1]
    train_metrics = Evaluation_results(new_train_y, xgb_train_preds, objective='train', num_features=num_features)
    print(train_metrics)

    # Predict on the testing set
    xgb_test_preds = xgb_model.predict(scaled_test_k_best)

    # Testing results
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
