import matplotlib
import numpy
import matplotlib.pyplot as plt

def plot(yData, Labels, xData, plotTitle, xTitle, yTitle):

    if len(yData) != len(Labels):
        print("Error: Data and Label mismatch")

    for i in range(len(yData)):
        if len(yData[i]) != len(xData[i]):
            print("Error: yData and xData mismatch")

    plt.clf()
    
    fig, ax = plt.subplots()
    ax.grid()
    ax.set(title=plotTitle)      
    ax.set(xlabel=xTitle, ylabel=yTitle)

    for i in range(len(yData)):
        ax.plot(xData[i], yData[i], label = Labels[i])
            
    ax.legend()

    plt.show()

def plot_confusion_matrix(cm, target_names, title='Confusion matrix'):
    
    # Confusion Matrix Plotter
    
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()