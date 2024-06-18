# train.py

# Importing important libraries
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
import joblib
import warnings

# Importing custom functions and configurations
from Functions import Remove_outliers_with_lof, Select_k_best_features, Evaluation_results
import config

warnings.filterwarnings('ignore')

def main():
    # Reading Data from Data Directory
    data_df = pd.read_csv(config.DATA_PATHS['train_data'])

    X_train = data_df.drop('critical_temp', axis=1)
    y_train = data_df['critical_temp']

 

    # Removing outliers
    new_train_X, new_train_y = Remove_outliers_with_lof(X_train, y_train, contamination=config.OUTLIER_REMOVAL_PARAMS['contamination'])

    # Selecting best Features
    k = config.FEATURE_SELECTION_PARAMS['k']
    X_train_filtered, scores = Select_k_best_features(new_train_X, new_train_y, k=k, score_func=mutual_info_regression)

    # Scaling Data
    selected_columns = X_train_filtered.columns
    scaler = StandardScaler()
    scaled_train_k_best = scaler.fit_transform(X_train_filtered)

    # Save the scaler for later use in test.py
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(selected_columns, 'selected_columns.pkl')

    # Final Model
    xgb_model = xgb.XGBRegressor(**config.MODEL_PARAMS)

    xgb_model.fit(scaled_train_k_best, new_train_y)
    xgb_train_preds = xgb_model.predict(scaled_train_k_best)

    # Training results
    print('XGBoost Results')
    num_features = scaled_train_k_best.shape[1]
    train_metrics = Evaluation_results(new_train_y, xgb_train_preds, objective='train', num_features=num_features)
    print(train_metrics)

    # Save the model
    joblib.dump(xgb_model, 'xgb_model.pkl')

if __name__ == "__main__":
    main()
