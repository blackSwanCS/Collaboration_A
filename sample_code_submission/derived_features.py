import pandas as pd

def derived_feature(df):
    # Perform calculations to derive features from the DataFrame
    # and store the results in new columns
    
    # Example: Derive a new feature by adding two existing features
    df['derived_feature'] = df['PRI_lep_pt'] + df['PRI_had_pt']
    
    
    # Return the DataFrame with the derived features
    return df