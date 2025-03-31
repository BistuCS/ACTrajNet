import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import argparse
from torch.utils.data import DataLoader


from model.CAF_tcn import ACTrajNet as CAF_tcn
from model.CAF_lstm import ACTrajNet as CAF_lstm
from model.CAF_bilstm import ACTrajNet as CAF_bilstm
from model.HCC_tcn import ACTrajNet as HCC_tcn
from model.HCC_lstm import ACTrajNet as HCC_lstm
from model.HCC_bilstm import ACTrajNet as HCC_bilstm
from model.VCC_tcn import ACTrajNet as VCC_tcn
from model.VCC_lstm import ACTrajNet as VCC_lstm
from model.VCC_bilstm import ACTrajNet as VCC_bilstm
from model.utils import ade, fde, mde

def main():
    parser = argparse.ArgumentParser(description='Visualize Aircraft Trajectories with Predictions')


    parser.add_argument('--data_file', type=str,
                        default='./dataset/7days1_no_social/test/11_0_0_fled.txt')
    parser.add_argument('--dataset_folder', type=str, default='/dataset/')
    parser.add_argument('--dataset_name', type=str, default='7days1_no_social')
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--obs', type=int, default=11)
    parser.add_argument('--preds', type=int, default=120)
    parser.add_argument('--preds_step', type=int, default=10)


    parser.add_argument('--model_type', type=str, default='CAF', choices=['CAF', 'HCC', 'VCC'])
    parser.add_argument('--encoder_type', type=str, default='tcn', choices=['tcn', 'lstm', 'bilstm'])


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


    parser.add_argument('--delim', type=str, default=' ')
    parser.add_argument('--model_dir', type=str, default="\\saved_models\\")
    parser.add_argument('--use_english', action='store_true',default=True)

    args = parser.parse_args()


    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False


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


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    data = pd.read_csv(args.data_file, header=None,
                       names=['time', 'id', 'x', 'y', 'z', 'windx', 'windy'],
                       sep='\s+')

    print(data.head())


    ModelClass = model_classes[args.model_type][args.encoder_type]
    model = ModelClass(args)
    model.to(device)


    model_suffix = f"_{args.model_type}_{args.encoder_type}"
    model_path = os.getcwd() + args.model_dir + "model_" + args.dataset_name + "_" + str(
        args.epoch) + model_suffix + ".pt"


    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()



    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')


    aircraft_ids = data['id'].unique()


    cmap_historic = plt.cm.Blues
    cmap_predicted = plt.cm.Reds
    cmap_ground_truth = plt.cm.Greens

    labels = {
        'hist': 'Historical Trajectory',
        'start': 'Starting Point',
        'gt': 'Ground Truth Future',
        'gt_end': 'Ground Truth Endpoint',
        'pred': 'Predicted Trajectory',
        'pred_end': 'Predicted Endpoint',
        'pred_points': 'Predicted Points',
        'gt_points': 'Ground Truth Points',
        'complete': 'Complete Trajectory',
        'title': '3D Visualization of Aircraft Trajectories (Historical, Predicted, and Ground Truth)'
    }

    for idx, aircraft_id in enumerate(aircraft_ids):
        aircraft_data = data[data['id'] == aircraft_id]


        aircraft_data = aircraft_data.sort_values(by='time')


        if len(aircraft_data) > args.obs:
            hist_data = aircraft_data.iloc[:args.obs]
            future_data = aircraft_data.iloc[args.obs:args.obs + args.preds]


            hist_x = hist_data['x'].values
            hist_y = hist_data['y'].values
            hist_z = hist_data['z'].values


            if not future_data.empty:

                sampled_indices = list(range(args.preds_step-1, len(future_data), args.preds_step))
                if sampled_indices:
                    gt_sampled = future_data.iloc[sampled_indices]
                    gt_x = gt_sampled['x'].values
                    gt_y = gt_sampled['y'].values
                    gt_z = gt_sampled['z'].values
                else:
                    gt_x, gt_y, gt_z = [], [], []
            else:
                gt_x, gt_y, gt_z = [], [], []


            ax.plot(hist_x, hist_y, hist_z,
                    color=cmap_historic(0.9),
                    linewidth=3,
                    label=labels['hist'] if idx == 0 else "")


            ax.scatter(hist_x[0], hist_y[0], hist_z[0],
                       color=cmap_historic(0.9),
                       s=50,
                       label=labels['start'] if idx == 0 else "")


            obs_traj = np.stack([hist_x, hist_y, hist_z], axis=1)  # [obs_len, 3]
            obs_traj = torch.FloatTensor(obs_traj).transpose(0, 1).unsqueeze(0)  # [1, 3, obs_len]


            wind_x = hist_data['windx'].values
            wind_y = hist_data['windy'].values
            context = np.stack([wind_x, wind_y], axis=1)  # [obs_len, 2]
            context = torch.FloatTensor(context).transpose(0, 1).unsqueeze(0)  # [1, 2, 11]


            adj = torch.ones((1, 1))


            z = torch.randn([1, 1, 128]).to(device)


            obs_traj = obs_traj.to(device)
            obs_traj = obs_traj.permute(2, 1, 0)  # [11, 3, 1]
            context = context.to(device)
            context = context.permute(2, 1, 0)  # [11, 2, 1]
            adj = adj.to(device)



            with torch.no_grad():
                pred_traj = model.inference(obs_traj, z, adj, context)


            pred_traj = pred_traj[0].cpu().numpy()


            pred_x = pred_traj[0,:]
            pred_y = pred_traj[1,:]
            pred_z = pred_traj[2,:]


            last_hist_x, last_hist_y, last_hist_z = hist_x[-1], hist_y[-1], hist_z[-1]


            pred_x_connected = np.concatenate(([last_hist_x], pred_x))
            pred_y_connected = np.concatenate(([last_hist_y], pred_y))
            pred_z_connected = np.concatenate(([last_hist_z], pred_z))


            ax.plot(pred_x_connected, pred_y_connected, pred_z_connected,
                    color=cmap_predicted(0.7),
                    linewidth=2, linestyle='--',
                    label=labels['pred'] if idx == 0 else "")


            ax.scatter(pred_x, pred_y, pred_z,
                       color=cmap_predicted(0.9),
                       s=80,
                       edgecolors='black',
                       linewidths=1,
                       label=labels['pred_points'] if idx == 0 else "")


            if len(gt_x) > 0:

                gt_x_connected = np.concatenate(([last_hist_x], gt_x))
                gt_y_connected = np.concatenate(([last_hist_y], gt_y))
                gt_z_connected = np.concatenate(([last_hist_z], gt_z))


                ax.plot(gt_x_connected, gt_y_connected, gt_z_connected,
                        color=cmap_ground_truth(0.7),
                        linewidth=2,
                        label=labels['gt'] if idx == 0 else "")


                ax.scatter(gt_x, gt_y, gt_z,
                           color=cmap_ground_truth(0.9),
                           s=80,
                           edgecolors='black',
                           linewidths=1,
                           label=labels['gt_points'] if idx == 0 else "")

            print(f"airclaft {aircraft_id} predict trajectory ")


            if len(gt_x) > 0:
                # Ensure pred and ground truth formats match what the utility functions expect
                # In utils.py, these functions expect arrays of shape (seq_len, dim)
                min_len = min(len(pred_x), len(gt_x))
                pred_coords = np.stack([pred_x[:min_len], pred_y[:min_len], pred_z[:min_len]], axis=1)
                gt_coords = np.stack([gt_x[:min_len], gt_y[:min_len], gt_z[:min_len]], axis=1)

                # Calculate metrics using the imported functions
                ade_value = ade(pred_coords, gt_coords)
                fde_value = fde(pred_coords, gt_coords)
                mde_value = mde(pred_coords, gt_coords)

                print(f" ADE: {ade_value:.6f}, FDE: {fde_value:.6f}, MDE: {mde_value:.6f}")
        else:

            print(f"airclaft {aircraft_id} point nums ({len(aircraft_data)}) less then obs_len ({args.obs})")
            x = aircraft_data['x'].values
            y = aircraft_data['y'].values
            z = aircraft_data['z'].values

            ax.plot(x, y, z,
                    color=cmap_historic(0.7),
                    linewidth=2,
                    label=f"{labels['complete']} ({aircraft_id})" if idx == 0 else "")
            ax.scatter(x[0], y[0], z[0],
                       color=cmap_historic(0.9),
                       s=100, edgecolors='k')


    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')


    handles, labels_list = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_list, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=10)


    ax.set_title(labels['title'])


    ax.view_init(elev=30, azim=45)


    plt.tight_layout()
    # plt.savefig('aircraft_trajectories.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()