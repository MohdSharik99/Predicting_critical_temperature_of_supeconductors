# importing important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def Basic_info_func(X):
    
    df = pd.DataFrame()
    df['Feature'] = X.columns
    df['Missing_values'] = X.isnull().sum().values
    df['N_uniques'] = X.nunique().values  
    df['Data_type'] = X.dtypes.values.astype(str)
    df['missing_percentage'] = (X.isnull().sum().values/len(X)).astype(int)
    
    
    return df.set_index('Feature')


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
    selected_features = scores.nlargest(k).index
    X_new = X[selected_features]

    # Plot the scores
    plt.figure(figsize=(10, 6))
    scores.nlargest(k).sort_values().plot(kind='barh')
    plt.title(f'Top {k} Features Selected by {score_func.__name__}')
    plt.xlabel('Score')
    plt.ylabel('Feature')
    plt.show()

    return X_new, scores


# PCA 

def Apply_pca(X, n_components=None, desired_variance=None):
    """
    Apply PCA to reduce dimensionality of the dataset based on number of components or desired variance.

    Parameters:
    - X: DataFrame or 2D array-like, containing features.
    - n_components: int or None, number of principal components to keep.
                    If None, all components are kept.
    - desired_variance: float or None, the amount of variance you want to retain.
                        Should be between 0 and 1.
                        If provided, overrides n_components.

    Returns:
    - X_pca: DataFrame containing the transformed features.
    - pca: The fitted PCA object.
    - n_components: int, the number of principal components retained.
    """
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=n_components)

    if desired_variance is not None:
        # If desired_variance is specified, calculate the number of components to retain
        pca.fit(X_scaled)
        cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_explained_variance >= desired_variance) + 1
        pca = PCA(n_components=n_components)

    # Fit and transform the data
    X_pca = pca.fit_transform(X_scaled)

    # Create DataFrame with principal components
    col_names = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    X_pca_df = pd.DataFrame(X_pca, columns=col_names)

    # Plot explained variance ratio
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio by Principal Components')
    plt.show()

    return X_pca_df, pca, n_components