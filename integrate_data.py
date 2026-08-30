import pandas as pd
import os

def integrate_datasets(data_path: str) -> None:
    # 1. load Walmart and FRED datasets
    walmart_df = pd.read_csv(f'Data/Walmart/Walmart_Sales.csv')

    fred_data_path = f'Data/FRED'
    fred_csv_names = [f for f in os.listdir(fred_data_path)]
    freds_df = {fred_csv_name.removesuffix('.csv'): 
                pd.read_csv(f'{fred_data_path}/{fred_csv_name}') 
                for fred_csv_name in fred_csv_names}

    # 2. Merge FREDs with Walmart (YearMonth basis)
    # The interval for all FREDs is on monthly basis, except for TDSP which is on quarterly basis
    walmart_df['Date'] = pd.to_datetime(walmart_df['Date'])
    walmart_df['YearMonth'] = walmart_df['Date'].dt.to_period("M")
    print(f'INFO >> Date converted to YearMonth type for <Walmart_Sales>.')

    for df_name, df in freds_df.items():
        if df_name != 'TDSP':
            df['index'] = pd.to_datetime(df['index'])
            df['YearMonth'] = df['index'].dt.to_period("M")
            
            # Drop date column, then drop duplicate YearMonth values
            df_clean = df.drop(columns=['index'])
            df_clean = df_clean.drop_duplicates(subset=['YearMonth'])

            # Merge with Walmart
            walmart_df = pd.merge(walmart_df, df_clean, on="YearMonth", how='left')
            print(f'INFO >> Date for <{df_name}> successfully cleaned and merged with <Walmart_Sales>.')

    # 3. Merge FREDs with Walmart (Quarter basis)
    tdsp_df = freds_df['TDSP']
    tdsp_df['Date'] = pd.to_datetime(tdsp_df['index'])
    tdsp_df.drop(columns=['index'], inplace=True)

    tdsp_df = tdsp_df.sort_values('Date')
    walmart_df = walmart_df.sort_values('Date')

    walmart_df = pd.merge_asof(walmart_df, tdsp_df, on='Date', direction='backward')
    # On sorted date, the values in the left are compared with a row on the right
    # if the left side >=  the righ side --> perfect, assign the value on the left to the right
    # o.w., move backward to the next value in the left df

    walmart_df.drop(columns=['YearMonth'], inplace=True)

    # Store the final dataframe in the designated path
    walmart_df.to_csv(f'{data_path}/inegrated_data.csv', index=False)


     


if __name__ == '__main__':
    integrate_datasets('Data')