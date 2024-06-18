import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


def Basic_info_func(X):
    
    df = pd.DataFrame()
    df['Feature'] = X.columns
    df['Missing_values'] = X.isnull().sum().values
    df['N_uniques'] = X.nunique().values  
    df['Data_type'] = X.dtypes.values.astype(str)
    df['missing_percentage'] = (X.isnull().sum().values/len(X)).astype(int)
    
    
    return df.set_index('Feature')

# outlier removal
def Remove_outliers_with_lof(train_X, train_y, contamination=0.05):
    """
    Detect outliers in the DataFrame using Local Outlier Factor (LOF) and remove them.
    
    Args:
    - train_X (DataFrame): DataFrame containing features.
    - train_y (Series): Target variable.
    - contamination (float): The proportion of outliers expected in the data.
    
    Returns:
    - new_train_X (DataFrame): DataFrame with outliers removed.
    - new_train_y (Series): Target variable corresponding to new_train_X.
    """
    # Concatenate train_X and train_y into a single DataFrame
    df = pd.concat([train_X, train_y], axis=1)
    
    # Print shape before outlier removal
    print("Shape before outlier removal:")
    print(df.shape)
    

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_X)

    # Apply Local Outlier Factor (LOF) to detect outliers
    lof = LocalOutlierFactor(contamination=contamination)
    outlier_preds = lof.fit_predict(X_scaled)

    # Identify indices of outliers
    outlier_indices = pd.Series(outlier_preds == -1, index=df.index)

    # Remove outliers from the DataFrame
    new_df = df[~outlier_indices].reset_index(drop=True)
    
    # Print shape after outlier removal
    print("\nShape after outlier removal:")
    print(new_df.shape)

    # Separate features and target variable in the new DataFrame
    new_train_X = new_df.drop(columns=train_y.name)
    new_train_y = new_df[train_y.name]

    return new_train_X, new_train_y


# calculate mutual information for feature selection

def Select_k_best_features(X, y, k=10, score_func=mutual_info_regression):
    """
    Select the k best features based on a scoring function.

    Parameters:
    - X: DataFrame or 2D array-like, containing features.
    - y: Series or 1D array-like, containing the target variable.
    - k: int, number of top features to select.
    - score_func: callable, scoring function to use (e.g., f_regression, mutual_info_regression).

    Returns:
    - X_new: DataFrame containing the k best features.
    - scores: Series containing the scores of the features.
    """
    # Apply SelectKBest with the given scoring function
    selector = SelectKBest(score_func=score_func, k=k)
    X_new = selector.fit_transform(X, y)
    
    # Get the scores of the features
    scores = pd.Series(selector.scores_, index=X.columns, name='Scores')
    scores = scores.nlargest(k).sort_values(ascending=False)
    selected_features = scores.nlargest(k).index
    X_new = X[selected_features]

#     # Plot the scores
#     plt.figure(figsize=(6, 5), dpi = 110)
#     ax = scores.plot(kind='barh')
#     plt.title(f'Top {k} Features Selected by {score_func.__name__}')
#     plt.xlabel('Score', size = 8)
#     plt.ylabel('Feature', size = 8)
    
#     # Annotate the bars with the feature scores
#     for index, value in enumerate(scores):
#         ax.text(value, index, f'{value:.2f}', va='center')
    
#     plt.show()

    return X_new, scores


   

# Adjusted r2_score
def Adjusted_r2_score(true_values, predicted_values, num_features):
    
    r2 = r2_score(true_values, predicted_values)

    n = len(true_values)
    
    p = num_features
    
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    
    return adjusted_r2

# Evaluation metrics
def Evaluation_results(true_values, predicted_values, objective='train', num_features=None):
    if objective == 'train':
        rmse_train = mean_squared_error(true_values, predicted_values, squared=False)
        mae_train = mean_absolute_error(true_values, predicted_values)
        r2_train = r2_score(true_values, predicted_values)
        adjusted_r2_train = Adjusted_r2_score(true_values, predicted_values, num_features)
        
        print('\n', '- '*30)
        print('Training results:')
        print(f'Training RMSE: {rmse_train:.5f}')
        print(f'Training MAE: {mae_train:.5f}')
        print(f'Training R2 score: {r2_train:.5f}')
        print(f'Training Adjusted R2 score: {adjusted_r2_train:.5f}')
        
        return ''
    
    elif objective == 'test':
        rmse_test = mean_squared_error(true_values, predicted_values, squared=False)
        mae_test = mean_absolute_error(true_values, predicted_values)
        r2_test = r2_score(true_values, predicted_values)
        adjusted_r2_test = Adjusted_r2_score(true_values, predicted_values, num_features)
        
        print('\n', '- '*30)
        print('\nTesting results:')
        print(f'Testing RMSE: {rmse_test:.5f}')
        print(f'Testing MAE: {mae_test:.5f}')
        print(f'Testing R2 score: {r2_test:.5f}')
        print(f'Testing Adjusted R2 score: {adjusted_r2_test:.5f}')
        
        return ''
    
    else:
        raise ValueError("Invalid value for 'objective'. Must be 'train' or 'test'.")
