import numpy as np
import tensorflow as tf

class WindowGenerator():
    def __init__(self, input_width, label_width,shift,
            train_df,val_df,label_columns=None):
        '''
        Initializiation of the WindowGenerator class that takes the following inputs
        and windows time series data into inputs and labels that can then be fed into
        a tensorflow model.
        ''' 
        # Store the dataframes
        self.train_df = train_df
        self.val_df= val_df

        # Grab label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_column_indices = {name: i for i, name in enumerate(label_columns)}

        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Develop window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice =  slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        '''
        This will split the windowed data into inputs and labels.
        '''
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        if self.label_columns is not None:
            labels = tf.stack(
                    [labels[:, :, self.column_indices[name]] for name in self.label_columns], axis =-1)

        inputs.set_shape([None, self.input_width,None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        '''
        Takes an array or dataframe and processes it into a dataset
        that can be read by tensorflow. Within thie function, the
        split_window is mapped to convert every window into inputs
        and labels.
        '''
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=self.total_window_size,
                sequence_stride=1,
                shuffle=True,
                batch_size=32,)

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        '''
        Turns the passed train dataframe into a dataset
        '''
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        '''
        Turns the passed validation dataframe into a dataset.
        '''
        return self.make_dataset(self.val_df)

    def __str__(self):
        '''
        Returns a summary of the intiliazed class WindowGenerator
        '''
        for example_inputs,example_labels in self.train.take(1):
            example_inputs_shape = example_inputs.shape
            example_labels_shape = example_labels.shape

        return '\n'.join([
            f'\n\nSummary of Dataset',
            f'-------------------',
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}',
            f'Inputs shape (batch,time,features): {example_inputs_shape}',
            f'Labels shape (batch,time,features); {example_labels_shape}'
            ])


