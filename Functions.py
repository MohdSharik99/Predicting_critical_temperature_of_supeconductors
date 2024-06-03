import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def Basic_info_func(X):
    
    df = pd.DataFrame()
    df['Feature'] = X.columns
    df['Missing_values'] = X.isnull().sum().values
    df['N_uniques'] = X.nunique().values  
    df['Data_type'] = X.dtypes.values.astype(str)
    df['missing_percentage'] = (X.isnull().sum().values/len(X)).astype(int)
    
    
    return df.set_index('Feature')
