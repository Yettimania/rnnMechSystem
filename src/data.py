import pandas as pd

def DataMetrics(dataframe):
    '''
    Returns information on dataframe to be analyzed.
    Input: Dataframe - Panda Dataframe
    Output: Null
    '''
    num_features = dataframe.shape[1]

    print( '\n'.join([
        f'Total features: {num_features}',
        f'File Features: {dataframe.columns.values}',
        f'Parametric Description',
        f'{dataframe.describe()}'
        ]))

def DataSplit(dataframe,train_split):
    '''
    Split the raw data into a train and validation set.
    Input:  Dataframe - Dataframe of data
            train_split - Percentage to Train
    Output: df_train - Dataframe containing training data
            df_val - Dataframe containing validation data
    '''
    print('\n\n\nSpltting the data set to test and train...')
    dataframe_len = len(dataframe)
    split_index = int(dataframe_len * train_split)
    df_train = dataframe[:split_index]
    df_val = dataframe[split_index:]
    print( '\n'.join([
        f'Total Length: {dataframe_len}',
        f'Split Index: {split_index}',
        f'Length of Train: {len(df_train)}',
        f'Length of Validation: {len(df_val)}'
        ]))
    return df_train,df_val

def AvgNorm(df_train,df_val):
    '''
    Normalize the datasets before feeding to neural network using the std and avg of parameters
    from the traiing data set
    Input: df_train - Training Dataset
           df_val - Valalidation Dataset
    Output:norm_train - Normalized Training Dataset
           norm_val - Normalzied Validation Dataset
    '''


    feature_names = df_train.columns.values
    
    df_norm_train = pd.DataFrame(columns = feature_names) 
    df_norm_val = pd.DataFrame(columns = feature_names) 

    feature_avg = list(map(lambda x: df_train[x].mean(), feature_names))
    feature_std = list(map(lambda x: df_train[x].std(), feature_names))

    index = 0
    for i in feature_names:
        train_slice = df_train.loc[:,i]
        val_slice = df_val.loc[:,i]
        df_norm_train[i] = ( train_slice - feature_avg[index]) / feature_std[index]
        df_norm_val[i] = ( val_slice - feature_avg[index]) / feature_std[index]
        index+=1
    
    print("\n\n\nReturning normalized train and validation set!")
    print("Normalized Training Dataset Preview:")
    print(df_norm_train.head(5))
    print("\nNormalized Validation Dataset Preview:")
    print(df_norm_val.head(5))

    return df_norm_train, df_norm_val
