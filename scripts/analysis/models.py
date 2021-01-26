import torch.nn as nn


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

# Genotyping models

class Genotyping_GRU(nn.Module):
    def __init__(self, n_hidden, n_layers, dropout, bidirectional):
        super(Genotyping_GRU, self).__init__()
        
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.gru = nn.GRU(         
            input_size = 10, # 10 features
            hidden_size = self.n_hidden,         
            num_layers = self.n_layers,           
            batch_first = True,
            dropout=self.dropout,
            bidirectional=self.bidirectional)
        
        if self.dropout != 0.0:
            self.dropout_layer = nn.Dropout(p=self.dropout)
        
        self.out = nn.Linear(self.n_hidden, 4) # 4 clases
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

class Genotyping_LSTM(nn.Module):
    def __init__(self, n_hidden, n_layers, dropout, bidirectional):
        super(Genotyping_LSTM, self).__init__()
        
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.LSTM = nn.LSTM(         
            input_size = 10, # 10 features
            hidden_size = self.n_hidden,         
            num_layers = self.n_layers,           
            batch_first = True,
            dropout=self.dropout,
            bidirectional=self.bidirectional)
        
        if self.dropout != 0.0:
            self.dropout_layer = nn.Dropout(p=self.dropout)
        
        self.out = nn.Linear(self.n_hidden, 4) # 4 classes
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

class Genotyping_RNN(nn.Module):
    def __init__(self, n_hidden, n_layers, dropout, bidirectional):
        super(Genotyping_RNN, self).__init__()
        
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.rnn = nn.RNN(         
            input_size = 10, # 10 features
            hidden_size = self.n_hidden,         
            num_layers = self.n_layers,           
            batch_first = True,
            dropout=self.dropout,
            bidirectional=self.bidirectional)
        
        if self.dropout != 0.0:
            self.dropout_layer = nn.Dropout(p=self.dropout)
        self.out = nn.Linear(self.n_hidden, 4) # 4 classes
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

class Genotyping_Transformer(nn.Module):
    def __init__(self, n_layers, dropout, seq_len):
        super(Genotyping_Transformer, self).__init__()

        self.n_layers = n_layers
        self.dropout = dropout
        self.seq_len = seq_len

        encoder_layer = nn.TransformerEncoderLayer(d_model=10, nhead=10, dropout=self.dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        if self.dropout != 0.0:
            self.dropout_layer = nn.Dropout(p=self.dropout)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(self.seq_len*10, 4)
        self.out_act = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.transformer(x)
        if self.dropout!= 0.0:
            out = self.dropout_layer(out)
        out = self.flatten(out)
        out = self.out(out)
        out = self.out_act(out)
        return out

class Genotyping_Perceptron(nn.Module):
    def __init__(self, seq_len):
        super(Genotyping_Perceptron, self).__init__()
        
        self.seq_len = seq_len
        
        self.out = nn.Linear(self.seq_len*10, 4)
        self.out_act = nn.Softmax(dim=1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        out = x.contiguous().view(batch_size, -1)
        out = self.out(out)
        out = self.out_act(out)
        return out