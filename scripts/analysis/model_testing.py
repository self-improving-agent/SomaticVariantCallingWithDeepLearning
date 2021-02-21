import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import os

from models import GRU, LSTM, RNN, Transformer, Perceptron


# Function for metrics calculation
def calculate_metrics(confusion_matrix):
    accuracy = []
    precision = []
    recall = []
    f1 = []
    
    for i in range(3):
        tp = confusion_matrix[i,i]
        fp = np.sum(confusion_matrix[:,i]) - tp
        fn = np.sum(confusion_matrix[i,:]) - tp
        tn = np.sum(confusion_matrix) - tp - fp - fn
        
        accuracy.append(round(100*(tp + tn) / (tn + fp + fn + tp),2))
        precision.append(round(100*(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,2))
        recall.append(round(100*(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,2))
        f1.append(round(2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0.0,2))
    
    return accuracy, precision, recall, f1

# Main function
def test_model(experiment_name, model_type, hidden_units, layers, dropout, bidirectional, test_x, test_y, path):

    # Set up GPU devide if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up data
    test_x = torch.Tensor(test_x).double().to(device)
    test_y = torch.Tensor(test_y).double()

    # Initialise model
    if model_type == "GRU":
        model = GRU(hidden_units, layers, dropout, bidirectional).double().to(device)
    elif model_type == "LSTM":
        model = LSTM(hidden_units, layers, dropout, bidirectional).double().to(device)
    elif model_type == "RNN":
        model = RNN(hidden_units, layers, dropout, bidirectional).double().to(device)
    elif model_type == "Transformer":
        model = Transformer(layers, dropout, test_x.shape[1]).double().to(device)
    elif model_type == "Perceptron":
        model = Perceptron(test_x.shape[1]).double().to(device)
    else:
        # Default to GRU
        model = GRU(hidden_units, layers, dropout, bidirectional).double().to(device)
 
    metrics_file = open("{}/tables/test/{}-test-metrics.txt".format(path, experiment_name), "w")

    saves = sorted(os.listdir("{}/models/checkpoints/{}".format(path,experiment_name[experiment_name.rfind('-')+1:])))
    nums = [int(save[save.rfind('-')+1:save.rfind('.')]) for save in saves]
    saves = [save[1] for save in sorted(zip(nums, saves))]

    # Iterate over saved checkpoints
    for save in saves:
        model.load_state_dict(torch.load("{}/models/checkpoints/{}/{}".format(path, experiment_name[experiment_name.rfind('-')+1:], save)))
        model.eval()

        # Run Inference
        pos = range(0, len(test_x), 250)
        test_output = np.zeros((len(test_x), 3))
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
        auc = []
        for i in range(3):
            auc.append(round(roc_auc_score(test_y[:,i], test_output[:,i]),4))

        # Calculate metrics
        test_accuracy, test_precision, test_recall, test_f1 = calculate_metrics(test_confusion_matrix)

        metrics_file.write("Epoch No\t{}\n".format(save[save.rfind('-')+1:-3])) 
        metrics_file.write("x\t\tGermline Variant\tSomatic Variant  Normal\n")
        metrics_file.write("Test Accuracy\t\t{}\t\t{}\t\t{}\n".format(test_accuracy[0], test_accuracy[1], test_accuracy[2]))
        metrics_file.write("Test Precision\t\t{}\t\t{}\t\t{}\n".format(test_precision[0], test_precision[1], test_precision[2]))
        metrics_file.write("Test Recall\t\t{}\t\t{}\t\t{}\n".format(test_recall[0], test_recall[1], test_recall[2]))
        metrics_file.write("Test F1 Score\t\t{}\t\t{}\t\t{}\n".format(test_f1[0], test_f1[1], test_f1[2]))
        metrics_file.write("Test AUCs\t\t{}\t\t{}\t\t{}\n".format(auc[0], auc[1], auc[2]))
        metrics_file.write("\n")

    # Calculate ROC stats
    fpr = dict()
    tpr = dict()

    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(test_y[:,i].cpu(), test_output[:,i])

    # Plot ROC curves
    plt.plot(fpr[0], tpr[0], '--', label="Germline Variant")
    plt.plot(fpr[1], tpr[1], '--', label="Somatic Variant")
    plt.plot(fpr[2], tpr[2], '--', label="Normal")
    x = np.linspace(0, 1, 2)
    plt.plot(x)
    plt.title("Validation ROC")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="best")
    plt.grid('on')
    plt.savefig("{}/figures/ROCs/{}_test_roc.pdf".format(path, experiment_name))
    plt.clf()