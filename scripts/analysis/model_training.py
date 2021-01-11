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
def train_model(experiment_name, model_type, epochs, learning_rate, batch_size, hidden_units, 
                layers, dropout, bidirectional, train_x, train_y, valid_x, valid_y, path):
    
    # Set up GPU devide if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up data
    train_x = torch.Tensor(train_x).double()
    train_y = torch.Tensor(train_y).double()

    valid_x = torch.Tensor(valid_x).double().to(device)
    valid_y = torch.Tensor(valid_y).double().to(device)

    train_data = Dataset(train_x, train_y)    
    train_data_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, num_workers=10)

    # Initialise model
    if model_type == "GRU":
        model = GRU(hidden_units, layers, dropout, bidirectional).double().to(device)
    elif model_type == "LSTM":
        model = LSTM(hidden_units, layers, dropout, bidirectional).double().to(device)
    elif model_type == "RNN":
        model = RNN(hidden_units, layers, dropout, bidirectional).double().to(device)
    elif model_type == "Transformer":
        model = Transformer(layers, dropout, train_x.shape[1]).double().to(device)
    elif model_type == "Perceptron":
        model = Perceptron(train_x.shape[1]).double().to(device)
    else:
        # Default to GRU
        model = GRU(hidden_units, layers, dropout, bidirectional).double().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)   
    loss_func = nn.CrossEntropyLoss()

    # Load and start from checkpoint if exists
    if os.path.exists("{}/tables/{}-metrics.txt".format(path, experiment_name)):
        start_epoch = int(sum(1 for line in open("{}/tables/{}-metrics.txt".format(path, experiment_name), "r")) / 6)
        metrics_file = open("{}/tables/{}-metrics.txt".format(path, experiment_name), "a")
        model.load_state_dict(torch.load("{}/models/{}-checkpoint.pt".format(path, experiment_name)))
    else:
        start_epoch = 0
        metrics_file = open("{}/tables/{}-metrics.txt".format(path, experiment_name), "w")

    # Model training
    for epoch in range(start_epoch, epochs):
        train_loss = 0.0
        train_confusion_matrix = np.zeros((3,3))
        
        # Training loop
        for step, (x, y) in enumerate(train_data_loader):        
            x, y = x.to(device), y.to(device)

            output = model(x)                               
            y = y.double()
            _, y = y.max(dim=1) 
            loss = loss_func(output.squeeze(), y)
            
            labels = np.argmax(output.detach().cpu().numpy(), axis=-1)

            class_missing = [num not in y for num in [0, 1, 2]]
            
            if any(class_missing):
                ind = [i for i in range(3) if class_missing[i]][0]
                cm = confusion_matrix(y.cpu(), labels)
                cm = np.insert(cm, ind, 0, axis=0)
                cm = np.insert(cm, ind, 0, axis=1)
                
                if cm.shape == (3,3):
                    train_confusion_matrix += cm
                else:
                    train_confusion_matrix += confusion_matrix(y.cpu(), labels)
            else:
                train_confusion_matrix += confusion_matrix(y.cpu(), labels)

            train_loss += loss.item()
            
            optimizer.zero_grad()                          
            loss.backward()                                 
            optimizer.step()
            
        train_loss /= (step+1)
        train_loss = np.round(train_loss,4)
        
        # Process Validation Set
        if model_type == "Transformer":
            # The transformer model needs too much memory, split validation into parts
            pos = range(0, len(valid_x), 250)
            valid_output = np.zeros((len(valid_x), 3))
            for i in range(1, len(pos)):
                start = pos[i-1]
                end = pos[i]
                
                current_part = valid_x[start:end]
                
                valid_output[start:end] = model(current_part).detach().cpu().numpy()
            valid_output = torch.from_numpy(valid_output)
        else:
            valid_output = model(valid_x).squeeze().detach().cpu()
        
        valid_output = valid_output.squeeze()
        _, valid_true_labels = valid_y.max(dim=1)
        valid_labels = np.argmax(valid_output, axis=-1)
        valid_loss = np.round(loss_func(valid_output, valid_true_labels).data.cpu().numpy(),4)
        valid_confusion_matrix = confusion_matrix(valid_true_labels.cpu(), valid_labels)

        # Calculate AUC
        auc = []
        for i in range(3):
            auc.append(round(roc_auc_score(valid_y[:,i], valid_output[:,i]),4))

        # Calculate metrics
        train_accuracy, train_precision, train_recall, train_f1 = calculate_metrics(train_confusion_matrix)
        valid_accuracy, valid_precision, valid_recall, valid_f1 = calculate_metrics(valid_confusion_matrix)

        # Save progress
        torch.save(model.state_dict(), "{}/models/{}-checkpoint.pt".format(path, experiment_name))

        # Print metrics to output
        metrics_file.write("Epoch No:\t{}\n".format(epoch+1))
        metrics_file.write("x\tAccuracy\tPrecision\tRecall\t\tF1 Score\tLoss\n")
        metrics_file.write("Train\t{}\t\t{}\t\t{}\t\t{}\t\t{}\n".format(train_accuracy, train_precision, train_recall, train_f1, train_loss))
        metrics_file.write("Valid\t{}\t\t{}\t\t{}\t\t{}\t\t{}\n".format(valid_accuracy, valid_precision, valid_recall, valid_f1, valid_loss))
        metrics_file.write("Class Valid AUCs:\t{}\t\t{}\t\t{}\n".format(auc[0], auc[1], auc[2]))
        metrics_file.write("\n")

        # Print progress
        print("Epoch no: {}/{}\t Train Accuracy: {}\t Validation Accuracy: {}\t Train loss: {}\t Validation Loss: {}\n".format(
            epoch+1, epochs, train_accuracy, valid_accuracy, train_loss, valid_loss))

    # Save model
    torch.save(model.state_dict(), "{}/models/{}.pt".format(path, experiment_name))

    # Calculate ROC stats
    fpr = dict()
    tpr = dict()

    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(valid_y[:,i].cpu(), valid_output[:,i])

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
    plt.savefig("{}/figures/ROCs/{}_valid_roc.pdf".format(path, experiment_name))
    plt.clf()