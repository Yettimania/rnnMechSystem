import matplotlib.pyplot as plt
import os

def plot_loss(history, title):
    '''
    Plot the loss for the inputed model and save to ./figs/ directory
    Input: history object returned from model
           title - "string" for file name
    '''
    plt.figure(figsize=(10,6))
    loss = history.history['loss']
    epochs = range(len(loss))
    plt.plot(epochs, loss, label=title)
    plt.title("Training Loss")
    plt.legend(["Loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    print('Saving loss image to /figs/...')
    print(os.path.join('./figs/',title + '.png'))
    plt.savefig(os.path.join('./figs',title))
