import pandas as pd

def derived_feature(df):
    # Perform calculations to derive features from the DataFrame
    # and store the results in new columns
    
    # Example: Derive a new feature by adding two existing features
    df['derived_feature'] = df['feature1'] + df['feature2']
    
    # Example: Derive another feature by subtracting two existing features
    df['derived_feature2'] = df['feature3'] - df['feature4']
    
    # Return the DataFrame with the derived features
    return df