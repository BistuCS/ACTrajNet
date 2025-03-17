import torch
import torch.nn as nn
import argparse
class LstmNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LstmNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)   
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # print("obs_traj shape is ", x.shape)

        # print('h0 shape', h0.shape)
        # print('c0 shape', c0.shape)
        out, _ = self.lstm(x, (h0, c0))
        # print('out shape',out.shape)
        # exit()
        out = torch.transpose(self.fc(out), 1, 2)
        return out

class BiLstmNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(BiLstmNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers * 2
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def init_hidden(self, batch):
        state0 = torch.zeros(self.num_layers, batch, self.h_dim)
        state1 = torch.zeros(self.num_layers, batch, self.h_dim)

        if self.use_cuda == 1:
            state0 = state0.cuda()
            state1 = state1.cuda()

        return (state0, state1)

    def forward(self, x):
      
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  #(4,1,256)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  #(4,1,256)
        
   
        out, _ = self.lstm(x, (h0, c0))

        return self.fc(out[:, -1, :]).unsqueeze(2)

class BiLstmNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(BiLstmNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers * 2
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def init_hidden(self, batch):
        state0 = torch.zeros(self.num_layers, batch, self.h_dim)
        state1 = torch.zeros(self.num_layers, batch, self.h_dim)

        if self.use_cuda == 1:
            state0 = state0.cuda()
            state1 = state1.cuda()

        return (state0, state1)

    def forward(self, x):
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        
        # print('h0 shape', h0.shape)
        # print('c0 shape', c0.shape)
        out, _ = self.lstm(x, (h0, c0))
        # print('out shape',out.shape)
        # exit()
        out = torch.transpose(self.fc(out), 1, 2)
        # print(out[:,-1,:].shape)
        return out

if __name__=='__main__':
    print(__name__)
    parser=argparse.ArgumentParser(description='Lstm')
    parser.add_argument('--lstm_input_channels', type=int, default=3)
    parser.add_argument('--lstm_hidden_size', type=int, default=256)
    parser.add_argument('--lstm_layers', type=int, default=2)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #
    # device = torch.device('cpu') #
    args=parser.parse_args()
    input_channels = args.lstm_input_channels
    n_classes = 12
    lstm_input_size=3
    lstm_hidden_size = args.lstm_hidden_size
    lstm_layers = args.lstm_layers
    dropout=0.5
    # lstm_encoder_x = BiLstmNet(input_size=lstm_input_size, hidden_size=lstm_hidden_size, output_size=n_classes , num_layers=lstm_layers, dropout=dropout).to(device)
    lstm_encoder_x = BiLstmNet(input_size=1, hidden_size=lstm_hidden_size, output_size=n_classes , num_layers=lstm_layers, dropout=dropout).to(device)
    x = torch.randn(1, 11, 1).to(device) # [batch_size, seq_size, channel_size]
    print(type(x), x.shape)
    y = lstm_encoder_x(x)
    sig = nn.Sigmoid()
    # print(y)
    print('lstm y shape', y.shape) # [1, 11, 11]
    print(sig(y).shape)
