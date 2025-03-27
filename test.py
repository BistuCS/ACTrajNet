import argparse
import os
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

# Import all model variants
from model.CAF_tcn import ACTrajNet as CAF_tcn
from model.CAF_lstm import ACTrajNet as CAF_lstm
from model.CAF_bilstm import ACTrajNet as CAF_bilstm
from model.HCC_tcn import ACTrajNet as HCC_tcn
from model.HCC_lstm import ACTrajNet as HCC_lstm
from model.HCC_bilstm import ACTrajNet as HCC_bilstm
from model.VCC_tcn import ACTrajNet as VCC_tcn
from model.VCC_lstm import ACTrajNet as VCC_lstm
from model.VCC_bilstm import ACTrajNet as VCC_bilstm

from model.utils import ade, fde, TrajectoryDataset, seq_collate, mde

def main():
    parser = argparse.ArgumentParser(description='Test Trajectory Prediction Models')
    
    # Dataset parameters
    parser.add_argument('--dataset_folder', type=str, default='/dataset/')
    parser.add_argument('--dataset_name', type=str, default='7days1_no_social')
    parser.add_argument('--epoch',type=int,required=True)
    parser.add_argument('--obs', type=int, default=11)
    parser.add_argument('--preds', type=int, default=120)
    parser.add_argument('--preds_step', type=int, default=10)
    
    # Model selection parameters
    parser.add_argument('--model_type', type=str, default='CAF', choices=['CAF', 'HCC', 'VCC'],
                        help='Model architecture type (CAF, HCC, VCC)')
    parser.add_argument('--encoder_type', type=str, default='tcn', choices=['tcn', 'lstm', 'bilstm'],
                        help='Encoder type (tcn, lstm, bilstm)')
    
    # Network parameters
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--tcn_channel_size', type=int, default=256)
    parser.add_argument('--tcn_layers', type=int, default=2)
    parser.add_argument('--tcn_kernels', type=int, default=4)
    parser.add_argument('--lstm_input_channels', type=int, default=1)
    parser.add_argument('--lstm_hidden_size', type=int, default=256)
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--num_context_input_c', type=int, default=2)
    parser.add_argument('--num_context_output_c', type=int, default=7)
    parser.add_argument('--cnn_kernels', type=int, default=2)
    parser.add_argument('--gat_heads', type=int, default=16)
    parser.add_argument('--graph_hidden', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--cvae_hidden', type=int, default=128)
    parser.add_argument('--cvae_channel_size', type=int, default=128)
    parser.add_argument('--cvae_layers', type=int, default=2)
    parser.add_argument('--mlp_layer', type=int, default=32)
    
    # Other parameters
    parser.add_argument('--delim', type=str, default=' ')
    parser.add_argument('--model_dir', type=str, default="\\saved_models\\")
    
    args = parser.parse_args()
    
    # Dictionary of model classes
    model_classes = {
        'CAF': {
            'tcn': CAF_tcn,
            'lstm': CAF_lstm,
            'bilstm': CAF_bilstm
        },
        'HCC': {
            'tcn': HCC_tcn,
            'lstm': HCC_lstm,
            'bilstm': HCC_bilstm
        },
        'VCC': {
            'tcn': VCC_tcn,
            'lstm': VCC_lstm,
            'bilstm': VCC_bilstm
        }
    }
    
    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"
    print(f"Loading Test Data from {datapath + 'test'}")
    
    dataset_test = TrajectoryDataset(
        datapath + "test", 
        obs_len=args.obs, 
        pred_len=args.preds, 
        step=args.preds_step, 
        delim=args.delim
    )
    
    loader_test = DataLoader(
        dataset_test,
        batch_size=1,
        num_workers=4,
        shuffle=True,
        collate_fn=seq_collate
    )
    
    # Instantiate the selected model
    ModelClass = model_classes[args.model_type][args.encoder_type]
    model = ModelClass(args)
    model.to(device)
    
    # Construct model path based on selected model type and encoder
    model_suffix = f"_{args.model_type}_{args.encoder_type}"
    model_path = os.getcwd() + args.model_dir + "model_" + args.dataset_name + "_" + str(args.epoch) + model_suffix + ".pt"
    
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test the model
    test_ade_loss, test_fde_loss, test_mde_loss = test(model, loader_test, device)
    
    print(f"Model: {args.model_type}_{args.encoder_type}")
    print(f"Epoch: {args.epoch}")
    print("Test ADE Loss: ",test_ade_loss,"Test FDE Loss: ",test_fde_loss,"Test MDE Loss", test_mde_loss)


def test(model, loader_test, device):
    tot_ade_loss = 0
    tot_fde_loss = 0
    tot_mde_loss = 0
    tot_batch = 0
    
    for batch in tqdm(loader_test, ncols=80):
        tot_batch += 1
        batch = [tensor.to(device) for tensor in batch]
        
        obs_traj_all, pred_traj_all, obs_traj_rel_all, pred_traj_rel_all, context, seq_start = batch
        num_agents = obs_traj_all.shape[1]
        
        best_ade_loss = float('inf')
        best_fde_loss = float('inf')
        best_mde_loss = float('inf')
        
        for i in range(5):
            z = torch.randn([1, 1, 128]).to(device)
            
            adj = torch.ones((num_agents, num_agents))
            recon_y_all = model.inference(torch.transpose(obs_traj_all, 1, 2), z, adj, torch.transpose(context, 1, 2))
            
            ade_loss = 0
            fde_loss = 0
            mde_loss = 0
            
            for agent in range(num_agents):
                obs_traj = np.squeeze(obs_traj_all[:, agent, :].cpu().numpy())
                pred_traj = np.squeeze(pred_traj_all[:, agent, :].cpu().numpy())
                recon_pred = np.squeeze(recon_y_all[agent].detach().cpu().numpy()).transpose()
                
                ade_loss += ade(recon_pred, pred_traj)
                fde_loss += fde(recon_pred, pred_traj)
                mde_loss += mde(recon_pred, pred_traj)
            
            ade_total_loss = ade_loss / num_agents
            fde_total_loss = fde_loss / num_agents
            mde_total_loss = mde_loss / num_agents
            
            if ade_total_loss < best_ade_loss:
                best_ade_loss = ade_total_loss
                best_fde_loss = fde_total_loss
                best_mde_loss = mde_total_loss
        
        tot_ade_loss += best_ade_loss
        tot_fde_loss += best_fde_loss
        tot_mde_loss += best_mde_loss
    
    return tot_ade_loss / tot_batch, tot_fde_loss / tot_batch, tot_mde_loss / tot_batch



if __name__ == '__main__':
    #args = argparse.ArgumentParser().parse_args()
    main()
