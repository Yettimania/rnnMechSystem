'''
Main script - rnn Mechanical Systems

This script will read a csv data file for a mechanical system into the pandas library. The data is split into training and test data. This dataframe is formed into a tensor which is fed into a RNN. The objective is to evaluate the 'Pressure' variable of the system which is dependent on ambient temperature and equipment flow rate.
'''

# Import libraries
import pandas as pd
from src.data import DataMetrics,DataSplit,AvgNorm


# Enter csv data path if different or change file name
csv_path = "./data/sys_data.csv"

# Read into a data frame
df = pd.read_csv(csv_path)

# Print metrics for raw data
DataMetrics(df)

# Split the data into 80-20 train test.
df_train, df_val = DataSplit(df,0.8)

# Normalize the data.
AvgNorm(df_train, df_val)
