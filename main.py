'''
Main script - rnn Mechanical Systems

This script will read a csv data file for a mechanical system into the pandas library. The data is split into training and test data. This dataframe is formed into a tensor which is fed into a RNN. The objective is to evaluate the 'Pressure' variable of the system which is dependent on ambient temperature and equipment flow rate.
'''

# Import libraries
import pandas as pd
from src.data import DataMetrics,DataSplit,AvgNormalization
from src.windowdata import WindowGenerator


# Enter csv data path if different or change file name
csv_path = "./data/sys_data.csv"

# Read into a data frame
df = pd.read_csv(csv_path)

# Print metrics for raw data
DataMetrics(df)

# Split the data into 80-20 train test.
split_index = 0.8
df_train, df_val = DataSplit(df,split_index)

# Normalize the data the training and val sets
train_df, val_df = AvgNormalization(df_train, df_val)

# Create windowed data sets and labels.
w1 = WindowGenerator(input_width=12,label_width=1,shift=1,train_df=train_df,val_df=val_df,
        label_columns=['Temp'])
print(w1)

# Print details on training dataset within WindowGenerator class
print("\n\nInspection of Training Element")
print(w1.train.element_spec)
print("\n\nExample of Label and Shapes")
for example_inputs, example_labels in w1.train.take(1):
    print(f'Inputs shape (batch,time,features): {example_inputs.shape}')
    print(f'Labels shape (batch,time,features): {example_labels.shape}')

# Print details on validation dataset within WindowGenerator class
print("\n\nInspection of Validation Element")
print(w1.val.element_spec)
print("\n\nExample of Label and Shapes")
for example_inputs, example_labels in w1.val.take(1):
    print(f'Inputs shape (batch,time,features): {example_inputs.shape}')
    print(f'Labels shape (batch,time,features): {example_labels.shape}')
