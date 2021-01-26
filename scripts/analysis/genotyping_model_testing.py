import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import os

from models import Genotyping_GRU, Genotyping_LSTM, Genotyping_RNN, Genotyping_Transformer, Genotyping_Perceptron


# Function for metrics calculation
def calculate_metrics(confusion_matrix):
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    
    for i in range(4):
        tp = confusion_matrix[i,i]
        fp = np.sum(confusion_matrix[:,i]) - tp
        fn = np.sum(confusion_matrix[i,:]) - tp
        tn = np.sum(confusion_matrix) - tp - fp - fn
        
        accuracy += 100*(tp + tn) / (tn + fp + fn + tp)
        precision += 100*(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall += 100*(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    
    accuracy = round(accuracy/4,2)
    precision = round(precision/4,2)
    recall = round(recall/4,2)   
    f1 = round(2 * precision * recall / (precision + recall),2)
    
    return accuracy, precision, recall, f1


# Main function
def genotyping_test_model(experiment_name, model_type, hidden_units, layers, dropout, bidirectional, test_x, test_y, path):

    # Set up GPU devide if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up data
    test_x = torch.Tensor(test_x).double().to(device)
    test_y = torch.Tensor(test_y).double()

    # Initialise model
    if model_type == "GRU":
        model = Genotyping_GRU(hidden_units, layers, dropout, bidirectional).double().to(device)
    elif model_type == "LSTM":
        model = Genotyping_LSTM(hidden_units, layers, dropout, bidirectional).double().to(device)
    elif model_type == "RNN":
        model = Genotyping_RNN(hidden_units, layers, dropout, bidirectional).double().to(device)
    elif model_type == "Transformer":
        model = Genotyping_Transformer(layers, dropout, test_x.shape[1]).double().to(device)
    elif model_type == "Perceptron":
        model = Genotyping_Perceptron(test_x.shape[1]).double().to(device)
    else:
        # Default to GRU
        model = Genotyping_GRU(hidden_units, layers, dropout, bidirectional).double().to(device)
 
    model.load_state_dict(torch.load("{}/models/{}.pt".format(path, experiment_name)))
    model.eval()

    # Run Inference
    pos = range(0, len(test_x), 250)
    test_output = np.zeros((len(test_x), 4))
    for i in range(1, len(pos)):
        start = pos[i-1]
        end = pos[i]
        
        current_part = test_x[start:end]
        
        test_output[start:end] = model(current_part).detach().cpu().numpy()
    test_output = torch.from_numpy(test_output)

    test_output = test_output.squeeze().detach().numpy()
    test_labels = np.argmax(test_output, axis=-1)
    _, test_true_labels = test_y.max(dim=1)
    test_confusion_matrix = confusion_matrix(test_true_labels, test_labels)

    # Calculate AUC
    auc = 0.0
    for i in range(4):
        auc += roc_auc_score(test_y[:,i], test_output[:,i])
    auc = round(auc/4, 4)

    # Calculate metrics
    test_accuracy, test_precision, test_recall, test_f1 = calculate_metrics(test_confusion_matrix)

    metrics_file = open("{}/tables/test/{}-test-metrics.txt".format(path, experiment_name), "w")
    metrics_file.write("Test Accuracy\t\t{}\n".format(test_accuracy))
    metrics_file.write("Test Precision\t{}\n".format(test_precision))
    metrics_file.write("Test Recall\t\t{}\n".format(test_recall))
    metrics_file.write("Test F1 Score\t\t{}\n".format(test_f1))
    metrics_file.write("Test AUC\t\t{}\n".format(auc))
    metrics_file.write("\n")

    # Calculate ROC stats
    fpr = dict()
    tpr = dict()

    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(test_y[:,i].cpu(), test_output[:,i])

    # Plot ROC curves
    plt.plot(fpr[0], tpr[0], '--', label="A")
    plt.plot(fpr[1], tpr[1], '--', label="T")
    plt.plot(fpr[2], tpr[2], '--', label="C")
    plt.plot(fpr[3], tpr[3], '--', label="G")
    x = np.linspace(0, 1, 2)
    plt.plot(x)
    plt.title("Validation ROC")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="best")
    plt.grid('on')
    plt.savefig("{}/figures/ROCs/{}_test_roc.pdf".format(path, experiment_name))
    plt.clf()