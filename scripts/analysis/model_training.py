import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import gc


EXPERIMENT_NAME = snakemake.config["model_name"]
MODEL_TYPE = snakemake.config["model_type"]

# Hyperparameters
EPOCHS = snakemake.config["epochs"]              
BATCH_SIZE = snakemake.config["batch_size"]     
LEARNING_RATE = snakemake.config["learning_rate"]
HIDDEN_UNITS = snakemake.config["hidden_units"]
LAYERS = snakemake.config["hidden_layers"]
DROPOUT = snakemake.config["dropout"] # Set to positive to include dropout layers
BIDIRECTIONAL = snakemake.config["bidirectional"] # Set to true to turn the models bi-directional

# Set up GPU devide if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set up data
data = np.load(snakemake.input[0])
labels = np.load(snakemake.input[1])
train_set_size = int(np.ceil(data.shape[0] * 0.9))

# Swap axes to have (Set_size x Seq_len x Features) size
train_x = data[:train_set_size]
train_x = np.swapaxes(train_x, -1, 1)
train_x = torch.Tensor(train_x).double()

train_y = labels[:train_set_size]
train_y = torch.Tensor(train_y).double()

valid_x = data[train_set_size:]
valid_x = np.swapaxes(valid_x, -1, 1)
valid_x = torch.Tensor(valid_x).double().to(device)

valid_y = labels[train_set_size:]
valid_y = torch.Tensor(valid_y).double().to(device)

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

train_data = Dataset(train_x, train_y)    
train_data_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, num_workers=10)


# Different possible models
class GRU(nn.Module):
    def __init__(self, n_hidden, n_layers, dropout, bidirectional):
        super(GRU, self).__init__()
        
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional

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
            self.bidirectional=bidirectional)
        
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

if MODEL_TYPE == "GRU":
    model = GRU(HIDDEN_UNITS, LAYERS, DROPOUT, BIDIRECTIONAL).double().to(device)
elif MODEL_TYPE == "LSTM":
    model = LSTM(HIDDEN_UNITS, LAYERS, DROPOUT, BIDIRECTIONAL).double().to(device)
elif MODEL_TYPE == "RNN":
    model = RNN(HIDDEN_UNITS, LAYERS, DROPOUT, BIDIRECTIONAL).double().to(device)
elif MODEL_TYPE == "Perceptron":
    model = Perceptron(train_x.shape[1]).double().to(device)
else:
    # Default to GRU
    model = GRU(HIDDEN_UNITS, LAYERS, DROPOUT, BIDIRECTIONAL).double().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)   
loss_func = nn.CrossEntropyLoss()

# Metrics to record
train_losses = []
valid_losses = []

# 0: Accuracy, 1: Precision, 2: Recall, 3: F1 Score
train_metrics = np.zeros((EPOCHS, 4))
valid_metrics = np.zeros((EPOCHS, 4))

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
        
    accuracy = round(accuracy/3,4)
    precision = round(precision/3,4)
    recall = round(recall/3,4)
    f1 = round(2 * precision * recall / (precision + recall),2)
    return np.array([accuracy, precision, recall, f1])

metrics_file = open(snakemake.output[0], "w")

# Model training
for epoch in range(EPOCHS):
    train_loss = 0.0
    train_confusion_matrix = np.zeros((3,3))
    
    # Training loop
    for step, (x, y) in enumerate(train_data_loader):        
        gc.collect()

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
    valid_output = model(valid_x)
    valid_output = valid_output.squeeze()
    _, valid_true_labels = valid_y.max(dim=1)
    valid_labels = np.argmax(valid_output.detach().cpu().numpy(), axis=-1)
    valid_loss = np.round(loss_func(valid_output, valid_true_labels).data.cpu().numpy(),4)
    valid_confusion_matrix = confusion_matrix(valid_true_labels.cpu(), valid_labels)
    
    # Calculate and store metrics
    train_metrics[epoch] = calculate_metrics(train_confusion_matrix)
    valid_metrics[epoch] = calculate_metrics(valid_confusion_matrix)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    # Print metrics to output
    metrics_file.write("Epoch No:\t{}\n".format(epoch+1))
    metrics_file.write("x\tAccuracy\tPrecision\tRecall\t\tF1 Score\tLoss\n")
    metrics_file.write("Train\t{:.2f}\t\t{:.2f}\t\t{:.2f}\t\t{:.2f}\t\t{}\n".format(train_metrics[epoch,0], train_metrics[epoch,1], train_metrics[epoch,2], train_metrics[epoch,3], train_loss))
    metrics_file.write("Valid\t{:.2f}\t\t{:.2f}\t\t{:.2f}\t\t{:.2f}\t\t{}\n".format(valid_metrics[epoch,0], valid_metrics[epoch,1], valid_metrics[epoch,2], valid_metrics[epoch,3], valid_loss))
    metrics_file.write("\n")

    # Free up memory
    del output, loss, valid_output

# Save model
torch.save(model.state_dict(), snakemake.output[1])

# Plot train metrics
plt.plot(train_metrics[1:,0], label="Accuracy")
plt.plot(train_metrics[1:,1], label="Precision")
plt.plot(train_metrics[1:,2], label="Recall")
plt.plot(train_metrics[1:,3], label="F1 Score")
plt.title("Training Metrics")
plt.xlabel("Epoch Number")
plt.ylabel("%")
plt.legend(loc="upper left")
plt.grid("on")
plt.savefig(snakemake.output[2])
plt.clf()


# Plot Validation metrics
plt.plot(valid_metrics[:,0], label="Accuracy")
plt.plot(valid_metrics[:,1], label="Precision")
plt.plot(valid_metrics[:,2], label="Recall")
plt.plot(valid_metrics[:,3], label="F1 Score")
plt.title("Validation Metrics")
plt.xlabel("Epoch Number")
plt.ylabel("%")
plt.legend(loc="upper left")
plt.grid("on")
plt.savefig(snakemake.output[3])
plt.clf()

# Plot losses
plt.plot(train_losses[1:], label="train_loss")
plt.plot(valid_losses[1:], label="valid_loss")
plt.title("Losses")
plt.xlabel("Epoch Number")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.grid("on")
plt.savefig(snakemake.output[4])
plt.clf()

# Calculate and plot ROC
valid_output = model(valid_x)
valid_output = valid_output.squeeze()

fpr = dict()
tpr = dict()
auc = dict()

for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(valid_y[:,i].cpu(), valid_output.detach().cpu().numpy()[:,i])
    auc[i] = roc_auc_score(valid_y[:,i].cpu(), valid_output.detach().cpu().numpy()[:,i])

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
plt.savefig(snakemake.output[5])
