import torch
from torch import nn

from model.tcn_model import TemporalConvNet
from model.gat_model import GAT
from model.cvae_base import CVAE
from model.utils import acc_to_abs
from model.lstm_model import LstmNet



class ACTrajNet(nn.Module):
    def __init__(self, args):
        super(ACTrajNet, self).__init__()

  

        input_size = args.input_channels

        n_classes = int(args.preds/args.preds_step)

        tcn_alti_out = 1
        num_channels_alt= [args.tcn_channel_size]*args.tcn_layers  #[256] * 2
        num_channels_alt.append(tcn_alti_out) # output's channel of tcn_alti 
        num_channels= [args.tcn_channel_size]*args.tcn_layers  #[256] * 2
        num_channels.append(n_classes)
        tcn_kernel_size = args.tcn_kernels
        dropout = args.dropout
        
        gat_in = n_classes*args.obs+args.num_context_output_c

        
        cvae_encoder = [n_classes*n_classes]
        for layer in range(args.cvae_layers):
            cvae_encoder.append(args.cvae_channel_size)
        cvae_decoder = [args.cvae_channel_size]*args.cvae_layers
        cvae_decoder.append(input_size*args.mlp_layer)


        self.tcn_encoder_altitude_X = TemporalConvNet(1, num_channels_alt, kernel_size=tcn_kernel_size, dropout=dropout)
        self.tcn_encoder_altitude_Y = TemporalConvNet(1, num_channels_alt, kernel_size=tcn_kernel_size, dropout=dropout)

        self.fc_a_X = nn.Linear(11, 12) 
        self.sig_a_X = nn.Sigmoid()
        self.sig_a_Y = nn.Sigmoid()
        self.fc_a_Y = nn.Linear(12, 12) 
        
        self.tcn_encoder_x = TemporalConvNet(input_size, num_channels, kernel_size=tcn_kernel_size, dropout=dropout)
        self.tcn_encoder_y = TemporalConvNet(input_size, num_channels, kernel_size=tcn_kernel_size, dropout=dropout)
        
        # self.cvae = CVAE(encoder_layer_sizes = cvae_encoder,latent_size = args.cvae_hidden, decoder_layer_sizes =cvae_decoder,conditional=True, num_labels= gat_out+gat_in)
        self.cvae = CVAE(encoder_layer_sizes = cvae_encoder,latent_size = args.cvae_hidden, decoder_layer_sizes =cvae_decoder,conditional=True, num_labels= gat_in)
        # self.gat = GAT( nin=gat_in, nhid = graph_hidden, nout = gat_out, alpha = alpha, nheads = n_heads)
        self.linear_decoder = nn.Linear(args.mlp_layer,n_classes)
        self.context_conv = nn.Conv1d(in_channels=args.num_context_input_c, out_channels=1, kernel_size=args.cnn_kernels)
        self.context_linear = nn.Linear(args.obs-1,args.num_context_output_c)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.linear_decoder.weight.data.normal_(0, 0.05)
        self.context_linear.weight.data.normal_(0, 0.05)
        self.context_conv.weight.data.normal_(0, 0.1)
        
    def forward(self, x, y, adj,context,sort=False):        
        #x [11,3,1]  #y [12,3,1]
        # encoded_trajectories_x = []
        encoded_appended_trajectories_x = []
        encoded_trajectories_y = []
        
        # pass all agents through encoder

        for agent in range(x.shape[2]):
            #x1.shape(1,3,11)
            x1 = torch.transpose(x[:,:, agent][None, :, :], 1, 2)

            
            # altitude_x1.shape(1,1,11)
            altitude_x1 = x[:, 2, agent][None, None, :]
            #encoded_alt_x.shape(1,1,11)
            encoded_alt_x = self.tcn_encoder_altitude_X(altitude_x1)

            #encoded_x1.shape(1,12,11)
            encoded_x1 = self.tcn_encoder_x(x1) 
            #encoded_x1.shape(1,12,11)
            encoded_x1 = encoded_x1 * torch.transpose(self.sig_a_X(self.fc_a_X(encoded_alt_x)), 1, 2)
            #encoded_x.shape(1,1,132)
            encoded_x = torch.flatten(encoded_x1)[None,None,:] # shape torch.Size([1, 1, 132])
            # encoded_trajectories_x.append(encoded_x)
            #c1.shape [1,2,11]
            c1 = torch.transpose(context[:,:, agent][None, :, :], 1, 2)
            #encoded_context [1,1,10]
            encoded_context = self.context_conv(c1)

            # encoded_context [1,1,7]
            encoded_context = self.relu(self.context_linear(encoded_context))
            

            
            #[1,1,139]
            appended_x = torch.cat((encoded_x, encoded_context),dim=2) # feature_dim_x + feature_dim_context

            
            encoded_appended_trajectories_x.append(appended_x)
            
            #y1[1,3,12]
            y1 = torch.transpose(y[:,:, agent][None, :, :], 1, 2)
            # altitude_y1.shape  [1, 1, 12]
            altitude_y1 = y[:, 2, agent][None, None, :] 
            encoded_alt_y = self.tcn_encoder_altitude_Y(altitude_y1)  # [1, 12, 12]
            # y1 shape: torch.Size([1, 3, 12])
            encoded_y1 = self.tcn_encoder_y(y1) # [1, 12, 12]
            encoded_y1 = encoded_y1 * torch.transpose(self.sig_a_Y(self.fc_a_Y(encoded_alt_y)), 1, 2)   #[1,12,12]
            # encoded_y2 = self.lstm_encoder_y(y[:,:, agent][None, :, :])
            # encoded_y2 shape: torch.Size([1, 12, 12])
            encoded_y = torch.flatten(encoded_y1)[None,None,:]   #[1,1,144]
            # appended_y = torch.cat((encoded_y, encoded_alt_y),dim=2) # 144 + 12
            appended_y = encoded_y
            encoded_trajectories_y.append(appended_y)



        recon_y = []
        m = []
        var = []
        
        # pass all agents through decoder
        for agent in range(x.shape[2]):
            
            H_xx = encoded_appended_trajectories_x[agent]

    

            H_y = encoded_trajectories_y[agent]
     
            H_yy, means, log_var, z = self.cvae(H_y,H_xx)
            # exit()
            H_yy =  torch.reshape(H_yy, (3, -1))
            recon_y_x = (self.linear_decoder(H_yy))
            recon_y_x = torch.unsqueeze(recon_y_x,dim=0)
            recon_y_x = acc_to_abs(recon_y_x,x[:,:,agent][:,:,None])    

            recon_y.append(recon_y_x)
            m.append(means)
            var.append(log_var)
        return recon_y,m,var
    
    
    def inference(self,x,z,adj,context):

        # encoded_trajectories_x = []
        encoded_appended_trajectories_x = []
        
        # pass all agents through encoder
        for agent in range(x.shape[2]):
            x1 = torch.transpose(x[:,:, agent][None, :, :], 1, 2)
            # x2 = x[:,:, agent][None, :, :]
            c1 = torch.transpose(context[:,:, agent][None, :, :], 1, 2)
            encoded_context = self.context_conv(c1)
            encoded_context = self.relu(self.context_linear(encoded_context))
            altitude_x = x[:, 2, agent][None, None, :]

            encoded_alt_x = self.tcn_encoder_altitude_X(altitude_x)
            encoded_x1 = self.tcn_encoder_x(x1)
            # encoded_x2 = self.lstm_encoder_x(x2)
            encoded_x1 = encoded_x1 * torch.transpose(self.sig_a_X(self.fc_a_X(encoded_alt_x)), 1, 2) 
            encoded_x = torch.flatten(encoded_x1)[None,None,:]

            # encoded_trajectories_x.append(encoded_x)
            appended_x = torch.cat((encoded_x, encoded_context),dim=2)

            encoded_appended_trajectories_x.append(appended_x)

        
        recon_y = []
        m = []
        var = []
        
        # pass all agents through decoder
        for agent in range(x.shape[2]):
   
            H_xx = encoded_appended_trajectories_x[agent]
        
            H_yy = self.cvae.inference(z,H_xx)
            H_yy = torch.reshape(H_yy, (3, -1))

            recon_y_x = (self.linear_decoder(H_yy)) 
            recon_y_x = torch.unsqueeze(recon_y_x,dim=0)
            recon_y_x = acc_to_abs(recon_y_x,x[:,:,agent][:,:,None])    

            recon_y.append(recon_y_x.squeeze().detach())
     
        return recon_y