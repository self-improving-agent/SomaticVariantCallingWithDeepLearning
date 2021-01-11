import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import os


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        X = self.data[index]
        y = self.labels[index]
        
        return X, y

# Different possible models
class GRU(nn.Module):
    def __init__(self, n_hidden, n_layers, dropout, bidirectional):
        super(GRU, self).__init__()
        
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.gru = nn.GRU(         
            input_size = 4, # 4 features: Normal REF %, Normal ALT %, Tumor REF %, Tumor ALT %
            hidden_size = self.n_hidden,         
            num_layers = self.n_layers,           
            batch_first = True,
            dropout=self.dropout,
            bidirectional=self.bidirectional)
        
        if self.dropout != 0.0:
            self.dropout_layer = nn.Dropout(p=self.dropout)
        
        self.out = nn.Linear(self.n_hidden, 3)
        self.out_act = nn.Softmax(dim=1)

    def forward(self, x):
        out, _ = self.gru(x)
        if self.bidirectional:
            out = out[:,:,:self.n_hidden] + out[:,:,self.n_hidden:]
        if out.shape[1] > 1:
            out = out[:,-1,:].squeeze()
        if self.dropout!= 0.0:
            out = self.dropout_layer(out)
        out = self.out(out)
        out = self.out_act(out)
        return out

class LSTM(nn.Module):
    def __init__(self, n_hidden, n_layers, dropout, bidirectional):
        super(LSTM, self).__init__()
        
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.LSTM = nn.LSTM(         
            input_size = 4, # 4 features: Normal REF %, Normal ALT %, Tumor REF %, Tumor ALT %
            hidden_size = self.n_hidden,         
            num_layers = self.n_layers,           
            batch_first = True,
            dropout=self.dropout,
            bidirectional=self.bidirectional)
        
        if self.dropout != 0.0:
            self.dropout_layer = nn.Dropout(p=self.dropout)
        
        self.out = nn.Linear(self.n_hidden, 3)
        self.out_act = nn.Softmax(dim=1)

    def forward(self, x):        
        out, _ = self.LSTM(x)
        if self.bidirectional:
            out = out[:,:,:self.n_hidden] + out[:,:,self.n_hidden:]
        if out.shape[1] > 1:
            out = out[:,-1,:].squeeze()
        if self.dropout!= 0.0:
            out = self.dropout_layer(out)
        out = self.out(out)
        out = self.out_act(out)
        return out

class RNN(nn.Module):
    def __init__(self, n_hidden, n_layers, dropout, bidirectional):
        super(RNN, self).__init__()
        
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.rnn = nn.RNN(         
            input_size = 4, # 4 features: Normal REF %, Normal ALT %, Tumor REF %, Tumor ALT %
            hidden_size = self.n_hidden,         
            num_layers = self.n_layers,           
            batch_first = True,
            dropout=self.dropout,
            bidirectional=self.bidirectional)
        
        if self.dropout != 0.0:
            self.dropout_layer = nn.Dropout(p=self.dropout)
        self.out = nn.Linear(self.n_hidden, 3)
        self.out_act = nn.Softmax(dim=1)

    def forward(self, x):
        out, _ = self.rnn(x)
        if self.bidirectional:
            out = out[:,:,:self.n_hidden] + out[:,:,self.n_hidden:]
        if out.shape[1] > 1:
            out = out[:,-1,:].squeeze()
        if self.dropout!= 0.0:
            out = self.dropout_layer(out)
        out = self.out(out)
        out = self.out_act(out)
        return out

class Transformer(nn.Module):
    def __init__(self, n_layers, dropout, seq_len):
        super(Transformer, self).__init__()

        self.n_layers = n_layers
        self.dropout = dropout
        self.seq_len = seq_len

        encoder_layer = nn.TransformerEncoderLayer(d_model=4, nhead=4, dropout=self.dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        if self.dropout != 0.0:
            self.dropout_layer = nn.Dropout(p=self.dropout)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(self.seq_len*4, 3)
        self.out_act = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.transformer(x)
        if self.dropout!= 0.0:
            out = self.dropout_layer(out)
        out = self.flatten(out)
        out = self.out(out)
        out = self.out_act(out)
        return out

class Perceptron(nn.Module):
    def __init__(self, seq_len):
        super(Perceptron, self).__init__()
        
        self.seq_len = seq_len
        
        self.out = nn.Linear(self.seq_len*4, 3)
        self.out_act = nn.Softmax(dim=1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        out = x.contiguous().view(batch_size, -1)
        out = self.out(out)
        out = self.out_act(out)
        return out

# Function for metrics calculation
def calculate_metrics(confusion_matrix):
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    
    for i in range(3):
        tp = confusion_matrix[i,i]
        fp = np.sum(confusion_matrix[:,i]) - tp
        fn = np.sum(confusion_matrix[i,:]) - tp
        tn = np.sum(confusion_matrix) - tp - fp - fn
        
        accuracy += 100*(tp + tn) / (tn + fp + fn + tp)
        precision += 100*(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall += 100*(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        
    accuracy = round(accuracy/3,2)
    precision = round(precision/3,2)
    recall = round(recall/3,2)
    f1 = round(2 * precision * recall / (precision + recall),2)
    return accuracy, precision, recall, f1

# Main function
def test_model(experiment_name, model_type, hidden_units, layers, dropout, bidirectional, test_x, test_y, path):

    # Set up GPU devide if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up data
    test_x = torch.Tensor(test_x).double()
    test_y = torch.Tensor(test_y).double()

    # Initialise model
    if model_type == "GRU":
        model = GRU(hidden_units, layers, dropout, bidirectional).double()
    elif model_type == "LSTM":
        model = LSTM(hidden_units, layers, dropout, bidirectional).double()
    elif model_type == "RNN":
        model = RNN(hidden_units, layers, dropout, bidirectional).double()
    elif model_type == "Transformer":
        model = Transformer(layers, dropout, test_x.shape[1]).double().to(device)
    elif model_type == "Perceptron":
        model = Perceptron(test_x.shape[1]).double()
    else:
        # Default to GRU
        model = GRU(hidden_units, layers, dropout, bidirectional).double()
 
    model.load_state_dict(torch.load("{}/models/{}.pt".format(path, experiment_name)))
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

    metrics_file = open("{}/tables/test/{}-test-metrics.txt".format(path, experiment_name), "w")
    metrics_file.write("x\tAccuracy\tPrecision\tRecall\t\tF1 Score\n")
    metrics_file.write("Test\t{}\t\t{}\t\t{}\t\t{}\n".format(test_accuracy, test_precision, test_recall, test_f1))
    metrics_file.write("Class Test AUCs:\t{}\t\t{}\t\t{}\n".format(auc[0], auc[1], auc[2]))
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