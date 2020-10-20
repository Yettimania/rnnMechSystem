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
           df_val - Validation Dataset
    Output:norm_train - Normalized Training Dataset
           norm_val - Normalzied Validation Dataset
    '''

