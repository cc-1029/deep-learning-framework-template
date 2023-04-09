import numpy as np
import pandas as pd

def reduce_df_memory(df):
    """
    Transform datasets type of dataframe's columns to save more memory

    Args:
        df: DataFrame to transform

    Returns:
        df: DataFrame which reduces memory
    """
    start_memory = df.memory_usage().sum() / 1024**3
    print(f'Memory usage before optimization is {start_memory:.3f} GB') 

    for col in df.columns:
        col_type = df[col].dtypes
        if str(col_type)[:3] == 'int':
            c_min, c_max = df[col].min(), df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df[col] = df[col].astype(np.int64)
        if str(col_type)[:5] == 'float':
            c_min, c_max = df[col].min(), df[col].max()
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            elif c_min > np.finfo(np.float64).min and c_max < np.finfo(np.float64).max:
                df[col] = df[col].astype(np.float64)
    
    end_memory = df.memory_usage().sum() / 1024**3
    print(f'Memory usage after optimization is {end_memory:.3f} GB')
    
    decrease_pp = 100 * (start_memory - end_memory) / start_memory
    print(f'Decreased by {decrease_pp:.2f}%')
    return df
