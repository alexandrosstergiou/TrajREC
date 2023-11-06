from functools import partial
import argparse
import numpy as np
import os
import torch
from utils import write_reconstructed_trajectories
from dataloader import load_evaluation_data

from trajectories import load_anomaly_masks
from utils import batch_inference, reconstruct_data, summarise_reconstruction
from models.trajrec import trajrec_tiny, trajrec_small, trajrec_base, trajrec_large, trajrec_huge, TrajREC



parser = argparse.ArgumentParser(description='Skeleton based anomaly detection.')
parser.add_argument('--gpu_id', default=0, type=int, help='Which GPUs to use. -1 for cpu')
parser.add_argument('--parallel', default=False, type=bool, help='Perform dataparallel training.')
parser.add_argument('--trajectories', type=str,
                       help='Path to directory containing training trajectories. For each video in the '
                            'training set, there must be a folder inside this directory containing the '
                            'trajectories.')
parser.add_argument('--video_resolution', default='856x480', type=str,
                       help='Resolution of the trajectories\' original video(s). It should be specified '
                            'as WxH, where W is the width and H the height of the video.')
parser.add_argument('--optimiser', default='adam', type=str, choices=['adam', 'rmsprop'],
                      help='Optimiser for model training.')
parser.add_argument('--learning_rate', default=0.001, type=float,
                      help='Learning rate of the optimiser.')
parser.add_argument('--loss', default='mse', type=str, choices=['log_loss', 'mae', 'mse'],
                      help='Loss function to be minimised by the optimiser.')
parser.add_argument('--epochs', default=5, type=int, help='Maximum number of epochs for training.')
parser.add_argument('--batch_size', default=256, type=int, help='Mini-batch size for model training.')
parser.add_argument('--coordinate_system', default='global', type=str,
                      choices=['global', 'bounding_box_top_left', 'bounding_box_centre'],
                      help='Which coordinate system to use.')
parser.add_argument('--normalisation_strategy', default='zero_one', type=str,
                      choices=['zero_one', 'three_stds', 'robust'],
                      help='Strategy for normalisation of the skeletons.')
parser.add_argument('--model', default='trajrec_tiny', type=str,help='Model architecture to use')
parser.add_argument('--extract_delta', action='store_true',
                            help='Only meaningful if model_type is global. If specified, include the difference '
                                 'between consecutive time-steps in addition to the absolute x and y coordinates.')
parser.add_argument('--input_length', default=12, type=int,
                            help='Number of input time-steps to encode.')
parser.add_argument('--reconstruct_reverse',type=bool, default=True,
                            help='Whether to reconstruct the reverse of the input sequence or not.')
parser.add_argument('--pred_length', default=6, type=int,
                             help='Number of time-steps to predict into future. Ignored if 0.')
parser.add_argument('--l1_reg', default=0.0, type=float,
                          help='Amount of L1 regularisation added to the model weights.')
parser.add_argument('--l2_reg', default=0.0, type=float,
                          help='Amount of L2 regularisation added to the model weights.')

parser.add_argument('--input_missing_steps', action='store_true',
                          help='Fill missing steps of trajectories with a weighted combination of '
                               'the closest non-missing steps.')
parser.add_argument('--reconstruct_original_data', type=bool, default=True,
                                    help='Whether to reconstruct the original trajectories or the concatenation '
                                         'of the output of the global and local models.')

parser.add_argument('--multiple_outputs', type=bool, default=True,
                                    help='If specified, the network also outputs the global and local '
                                         'reconstructions/predictions.')
parser.add_argument('--rec_length', default=12, type=int,
                                    help='Number of time-steps to decode from the input sequence.')
parser.add_argument('--global_normalisation_strategy', default='zero_one', type=str,
                                  choices=['zero_one', 'three_stds', 'robust'],
                                  help='Global normalisation strategy.')
parser.add_argument('--local_normalisation_strategy', default='zero_one', type=str,
                                  choices=['zero_one', 'three_stds', 'robust'],
                                  help='Local normalisation strategy.')
parser.add_argument('--out_normalisation_strategy', default='zero_one', type=str,
                                  choices=['zero_one', 'three_stds', 'robust'])
parser.add_argument('--chkp', type=str, help='Path to the checkpoint for loading the pretrained weights')
parser.add_argument('--write_reconstructions', action='store_true')
parser.add_argument('--write_predictions', action='store_true')
parser.add_argument('--write_predictions_bounding_boxes', action='store_true')
parser.add_argument('--write_bounding_boxes', action='store_true')
parser.add_argument('--lambda1', default=3.0, type=float)
parser.add_argument('--lambda2', default=3.0, type=float)
parser.add_argument('--lambda3', default=5.0, type=float)





def main():
    args = parser.parse_args()
    args = vars(args)
    device = torch.device(args['gpu_id'] if args['gpu_id'] != -1 else "cpu")
    global_input_dim = 4
    local_input_dim = 34
    
    if 'trajrec' in args['model']:
        if 'tiny' in args['model'] :
            model = trajrec_tiny(input_length=args['input_length'], global_input_dim=global_input_dim,
                    local_input_dim=local_input_dim, prediction_length=args['pred_length'], lambdas=[args['lambda1'],args['lambda2'],args['lambda3']])
        elif 'small' in args['model'] :
            model = trajrec_small(input_length=args['input_length'], global_input_dim=global_input_dim,
                    local_input_dim=local_input_dim, prediction_length=args['pred_length'], lambdas=[args['lambda1'],args['lambda2'],args['lambda3']])
        elif 'base' in args['model'] :
            model = trajrec_base(input_length=args['input_length'], global_input_dim=global_input_dim,
                    local_input_dim=local_input_dim, prediction_length=args['pred_length'], lambdas=[args['lambda1'],args['lambda2'],args['lambda3']])
        elif 'large' in args['model'] :
            model = trajrec_large(input_length=args['input_length'], global_input_dim=global_input_dim,
                    local_input_dim=local_input_dim, prediction_length=args['pred_length'], lambdas=[args['lambda1'],args['lambda2'],args['lambda3']])
        elif 'huge' in args['model'] :
            model = trajrec_huge(input_length=args['input_length'], global_input_dim=global_input_dim,
                    local_input_dim=local_input_dim, prediction_length=args['pred_length'], lambdas=[args['lambda1'],args['lambda2'],args['lambda3']])
        elif 'custom' in args['model'] :
            model =  TrajREC(embed_dim=args['embed_dim'], depth=args['depth'], num_heads=args['num_heads'], decoder_embed_dim=args['decoder_embed_dim'], decoder_depth=args['decoder_depth'], decoder_num_heads=args['decoder_num_heads'], mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),input_length=args['input_length'], global_input_dim=global_input_dim, local_input_dim=local_input_dim, prediction_length=args['pred_length'], lambdas=[args['lambda1'],args['lambda2'],args['lambda3']])
    else:
        raise ValueError(f"Invalid model {args['model']}")
    
    checkpoint = torch.load(args['chkp'], map_location=device)
    
    
    if args['chkp']:
        model.load_state_dict(checkpoint["model"])
        print(f"Loaded pretrained weights for the model")
    
    bb_scaler = checkpoint['bb_scaler']
    joint_scaler = checkpoint['joint_scaler']
    out_scaler = checkpoint['out_scaler']
    
    model = model.to(device)
    model.eval()
    
    input_length = model.input_length
    pred_length = model.prediction_length
    all_trajectories_path = os.path.join(args['trajectories'], 'trajectories')
    all_anomaly_masks = os.path.join(args['trajectories'], 'frame_level_masks')
    input_length = args['input_length']
    pred_length = args['pred_length']
    video_resolution = args['video_resolution']
    sort='avenue' in args['trajectories'].lower()
    
    video_resolution = [int(measurement) for measurement in video_resolution.split('x')]
    video_resolution = np.array(video_resolution, dtype=np.float32)
    data = []
    for camera_id in sorted(os.listdir(all_trajectories_path)):
        trajectories_path = os.path.join(all_trajectories_path, camera_id)
        anomaly_masks = load_anomaly_masks(os.path.join(all_anomaly_masks, camera_id))
        trajectories_ids, frames, X_global, X_local, X_out, _, _, _ = \
            load_evaluation_data(bb_scaler, joint_scaler, out_scaler, trajectories_path, input_length, 0, pred_length,
                                 video_resolution, 'zero_one', 'zero_one', 'zero_one', True, sort)
        data.append((anomaly_masks, trajectories_ids, frames, X_global, X_local, X_out))
    
    settings = ['past','present','future']
    
    for setting in settings:
        for anomaly_masks, trajectories_ids, frames, X_global, X_local, X_out in data:
            with torch.no_grad():
                predicted_frames = frames[:, :pred_length] + input_length
                predicted_ids = trajectories_ids[:, :pred_length]
                
                out, _ = batch_inference(model, [X_global, X_local, X_out], batch_size=1024, setting=setting)
                _, _, predicted_out = out
                
                if setting=='past':
                    predicted_out = predicted_out[:,:pred_length]
                elif setting=='future':
                    predicted_out = predicted_out[:,input_length:]
                else:
                    predicted_out = predicted_out[:,input_length//2:input_length//2+pred_length]
                
                predicted_y_traj = reconstruct_data(predicted_out, video_resolution, args['reconstruct_original_data'],
                                                bb_scaler, joint_scaler, out_scaler)

                prediction_ids, prediction_frames, predicted_y_traj = \
                summarise_reconstruction(predicted_y_traj, predicted_frames, predicted_ids)
                
                write_reconstructed_trajectories('reconstructed', predicted_y_traj, prediction_ids, prediction_frames,
                                                trajectory_type=f'predicted_skeleton_{setting}')
        
        


if __name__ == '__main__':
    main()
