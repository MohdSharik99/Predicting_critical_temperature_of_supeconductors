# config.py

# Data paths
DATA_PATHS = {
    'train_data': './Data/train_df.csv',
    'test_data': './Data/test_df.csv',
    'data': './Data/Source_data.csv'  # Adjust the path according to your actual file structure
}

# Model parameters
MODEL_PARAMS = {
    'objective': 'reg:squarederror',
    'n_estimators': 400,
    'learning_rate': 0.1,
    'max_depth': 10,
    'reg_lambda': 0.6,
    'reg_alpha': 0.05,
    'random_state': 42
}

# Feature selection parameters
FEATURE_SELECTION_PARAMS = {
    'k': 30,
    'score_func': 'mutual_info_regression'
}

# Outlier removal parameters
OUTLIER_REMOVAL_PARAMS = {
    'contamination': 0.005
}

# Plot parameters
PLOT_PARAMS = {
    'style': 'darkgrid',
    'xlabel': 'Actual',
    'ylabel': 'Predicted',
    'title': 'Actual vs Predicted Critical Temperature'
}
