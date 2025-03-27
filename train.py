import argparse
import os 
from tqdm import tqdm 
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim

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

from model.utils import TrajectoryDataset, seq_collate, loss_func,ade,fde,mde


# torch.backends.cudnn.enabled = False

def train():
    ##Dataset params
    parser = argparse.ArgumentParser(description='Train ACTrajNet model')
    parser.add_argument('--dataset_folder', type=str, default='/dataset/')
    parser.add_argument('--dataset_name', type=str, default='7days1_no_social')
    parser.add_argument('--obs', type=int, default=11)
    parser.add_argument('--preds', type=int, default=120)
    parser.add_argument('--preds_step', type=int, default=10)

    ##Network params
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--tcn_channel_size', type=int, default=256)
    parser.add_argument('--tcn_layers', type=int, default=2)
    parser.add_argument('--tcn_kernels', type=int, default=4)

    # LSTM module parameters
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

    parser.add_argument('--lr', type=float, default=0.00001)

    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--resume', type=str, default='', help='resume from checkpoint')
    parser.add_argument('--total_epochs', type=int, default=50)
    parser.add_argument('--delim', type=str, default=' ')
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--model_pth', type=str, default="/saved_models/")
    
    # Add model architecture and type arguments
    parser.add_argument('--model_arch', type=str, default='CAF', choices=['CAF', 'HCC', 'VCC'], 
                        help='Model architecture: CAF or HCC or VCC')
    parser.add_argument('--model_type', type=str, default='tcn', choices=['tcn', 'lstm', 'bilstm'], 
                        help='Model type: tcn, lstm, or bilstm')

    args = parser.parse_args()
    print(args)
    
    ##Select device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device",device)
    
    ##Load test and train data
    datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"

    print("Loading Train Data from", datapath + "train")
    dataset_train = TrajectoryDataset(datapath + "train", obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim)

    print("Loading Test Data from", datapath + "test")
    dataset_test = TrajectoryDataset(datapath + "test", obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim)

    loader_train = DataLoader(dataset_train, batch_size=1, num_workers=4, shuffle=True, collate_fn=seq_collate)
    loader_test = DataLoader(dataset_test, batch_size=1, num_workers=4, shuffle=True, collate_fn=seq_collate)

    # Initialize model based on architecture and type arguments
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

    # Select appropriate model class
    model_class = model_classes[args.model_arch][args.model_type]
    model = model_class(args)
    model.to(device)
    
    # Create model suffix for saving
    model_suffix = f'_{args.model_arch}_{args.model_type}'

    ##Resume
    if args.resume:
        checkpoint = torch.load(os.getcwd() + '/saved_models/' + args.resume, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        args.start_epoch = checkpoint['epoch'] + 1

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

 
    print(f"Starting Training with {args.model_arch}-{args.model_type} model...")

    for epoch in range(args.start_epoch, args.total_epochs + 1):
        model.train()
        loss_batch = 0 
        batch_count = 0
        tot_batch_count = 0
        tot_loss = 0
        for batch in tqdm(loader_train, ncols=80):
            batch_count += 1
            tot_batch_count += 1
            batch = [tensor.to(device) for tensor in batch]
            obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start = batch 
            num_agents = obs_traj.shape[1]
            pred_traj = torch.transpose(pred_traj, 1, 2)
            adj = torch.ones((num_agents, num_agents))

            optimizer.zero_grad()
            recon_y, m, var = model(torch.transpose(obs_traj, 1, 2), pred_traj, adj[0], torch.transpose(context, 1, 2))
            loss = 0
            
            for agent in range(num_agents):
                loss += loss_func(recon_y[agent], torch.transpose(pred_traj[:, :, agent], 0, 1).unsqueeze(0), m[agent], var[agent])
            
            loss_batch += loss
            tot_loss += loss.item()
            if batch_count > 8:
                loss_batch.backward()
                optimizer.step()
                loss_batch = 0 
                batch_count = 0

        print("EPOCH:", epoch, "Train Loss:", loss)

        if args.save_model:  
            loss = tot_loss / tot_batch_count
            model_path = os.getcwd() + args.model_pth + "model_" + args.dataset_name + "_" + str(epoch) + model_suffix + ".pt"
            print("Saving model at", model_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, model_path)
        
        # Evaluate model if requested and epoch is high enough
        # The HCC models had 45 as the threshold, keeping it consistent for all models
        if args.evaluate and epoch >= 45:
            model.eval()
            test_ade_loss, test_fde_loss, test_mde_loss = test(model, loader_test, device)
            print("EPOCH:", epoch, "Train Loss:", loss, "Test ADE Loss:", test_ade_loss, 
                  "Test FDE Loss:", test_fde_loss, "Test MDE Loss:", test_mde_loss)
    
    print("tot_batch_count", tot_batch_count)

def test(model,loader_test,device):
    tot_ade_loss = 0
    tot_fde_loss = 0
    tot_mde_loss = 0
    tot_batch = 0
    for batch in tqdm(loader_test, ncols=80):
        tot_batch += 1
        batch = [tensor.to(device) for tensor in batch]

        obs_traj_all , pred_traj_all, obs_traj_rel_all, pred_traj_rel_all, context, seq_start  = batch
        num_agents = obs_traj_all.shape[1]
        
        best_ade_loss = float('inf')
        best_fde_loss = float('inf')
        best_mde_loss = float('inf')

        for i in range(5):
            z = torch.randn([1,1 ,128]).to(device)
            
            adj = torch.ones((num_agents,num_agents))
            recon_y_all = model.inference(torch.transpose(obs_traj_all,1,2),z,adj,torch.transpose(context,1,2))
            
            ade_loss = 0
            fde_loss = 0
            mde_loss = 0
            for agent in range(num_agents):
                obs_traj = np.squeeze(obs_traj_all[:,agent,:].cpu().numpy())
                pred_traj = np.squeeze(pred_traj_all[:,agent,:].cpu().numpy())
                recon_pred = np.squeeze(recon_y_all[agent].detach().cpu().numpy()).transpose()
                ade_loss += ade(recon_pred, pred_traj)
                fde_loss += fde((recon_pred), (pred_traj))
                mde_loss += mde((recon_pred), (pred_traj))
           
            
            ade_total_loss = ade_loss/num_agents
            fde_total_loss = fde_loss/num_agents
            mde_total_loss = mde_loss/num_agents
            if ade_total_loss<best_ade_loss:
                best_ade_loss = ade_total_loss
                best_fde_loss = fde_total_loss
                best_mde_loss = mde_total_loss

        tot_ade_loss += best_ade_loss
        tot_fde_loss += best_fde_loss
        tot_mde_loss += best_mde_loss
    return tot_ade_loss/(tot_batch),tot_fde_loss/(tot_batch),tot_mde_loss/(tot_batch)

if __name__=='__main__':
    train()