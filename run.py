from functools import partial
import argparse
import datetime
import itertools
import math
import os
import pickle
import random
import sys
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.data import TensorDataset
from tqdm import tqdm
import utils
from sklearn.metrics import roc_auc_score

from dataloader import create_train_val_v2, _construct_output_data_alt, load_evaluation_data
from trajectories import assemble_ground_truth_and_reconstructions, load_anomaly_masks, compute_rnn_ae_reconstruction_errors, summarise_reconstruction_errors, discard_information_from_padded_frames
from utils import batch_inference

from models.trajrec import trajrec_tiny, trajrec_small, trajrec_base, trajrec_large, trajrec_huge, TrajREC

import wandb



@torch.no_grad()
def prediction_auc_score(model, data, reconstruct_original_data=True, batch_size=None, setting='future', is_avenue=False):
    input_length = model.input_length
    pred_length = model.prediction_length

    all_y_true, all_y_hat = [], []
    all_y_grouped_true, all_y_grouped_hat = {}, {}
    
    for anomaly_masks, trajectories_ids, frames, X_global, X_local, X_out in data:
        predicted_frames = frames[:, :pred_length] + input_length
        predicted_ids = trajectories_ids[:, :pred_length]
        
        out, target = batch_inference(model, [X_global, X_local, X_out], batch_size=batch_size, setting=setting)
        predicted_global, predicted_local, predicted_out = out
        
        X = X_out if reconstruct_original_data else np.concatenate((X_global, X_local), axis=-1)
        #y = retrieve_future_skeletons(trajectories_ids, X, pred_length)
        y = target[-1]
        predicted_y = predicted_out if reconstruct_original_data else np.concatenate((predicted_global, predicted_local), axis=-1)

        pred_errors = compute_rnn_ae_reconstruction_errors(y, predicted_y, 'mse')
        
        if setting=='past':
            pred_errors = pred_errors[:,:pred_length]
        elif setting=='future':
            pred_errors = pred_errors[:,input_length:]
        else:
            pred_errors = pred_errors[:,input_length//2:input_length//2+pred_length]
                    
        pred_ids, pred_frames, pred_errors = discard_information_from_padded_frames(predicted_ids, predicted_frames,
                                                                                    pred_errors, pred_length)
        pred_ids, pred_frames, pred_errors = summarise_reconstruction_errors(pred_errors, pred_frames, pred_ids)
        y_true_pred, y_hat_pred, y_grouped_true, y_grouped_hat = assemble_ground_truth_and_reconstructions(
                anomaly_masks, pred_ids, pred_frames, pred_errors, return_grouped_scores=True)
        all_y_true.append(y_true_pred)
        all_y_hat.append(y_hat_pred)
        all_y_grouped_true.update(y_grouped_true)
        all_y_grouped_hat.update(y_grouped_hat)
    all_y_true = np.concatenate(all_y_true)
    all_y_hat = np.concatenate(all_y_hat)
    
    if is_avenue:
        with open('data/masked_frames.pkl', 'rb') as f:
            AVENUE_MASK = pickle.load(f)
        all_y_true = all_y_true[AVENUE_MASK]
        all_y_hat = all_y_hat[AVENUE_MASK]
    
    scores = []
    sz = len(all_y_hat)

    mask = [1 for _ in range(len(all_y_true))]
    for i,m in enumerate(mask):
        if i in scores:
            mask[i] = 0
    mask = [m == 1 for m in mask]
    
    return roc_auc_score(all_y_true, all_y_hat), all_y_grouped_true, all_y_grouped_hat


def create_train_val_datasets(args):
    x_train, y_train, val_data, train_trajectories, val_trajectories, bb_scaler, joint_scaler, out_scaler = \
            create_train_val_v2(trajectories_path=args['trajectories'], video_resolution=args['video_resolution'],
                                input_length=args['input_length'], pred_length=args['pred_length'])

    x_global_train, x_local_train, x_out_train = x_train
    x_global_val, x_local_val, x_out_val = val_data[0]

    if y_train is not None:  # yes
        y_global_train, y_local_train, y_out_train = y_train
        y_global_val, y_local_val, y_out_val = val_data[1]
    else:
        y_global_train = y_local_train = y_out_train = y_global_val = y_local_val = y_out_val = None

    y = _construct_output_data_alt(True, args['rec_length'], args['reconstruct_reverse'],
                                   args['pred_length'], x_out_train, y_out_train,
                                   x_global_train, y_global_train, x_local_train, y_local_train)
    y_val = _construct_output_data_alt(True, args['rec_length'], args['reconstruct_reverse'],
                                       args['pred_length'], x_out_val, y_out_val,
                                       x_global_val, y_global_val, x_local_val, y_local_val)

    X = (x_global_train, x_local_train, x_out_train)
    X_val = (x_global_val, x_local_val, x_out_val)

    train_tensors = [torch.from_numpy(d) for d in itertools.chain(X, (a.copy() for a in y))]
    val_tensors = [torch.from_numpy(d) for d in itertools.chain(X_val, (a.copy() for a in y_val))]
    
    return x_local_train.shape[-1], TensorDataset(*train_tensors), TensorDataset(*val_tensors), \
            bb_scaler, joint_scaler, out_scaler


def run(args):
    print(args)
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    
    if 'avenue' in args['trajectories'].lower():
        project = "trajectory-anomalies-ave"
    elif 'shanghaitech' in args['trajectories'].lower():
        project = "trajectory-anomalies-stc"
    elif 'ubnormal' in args['trajectories'].lower():
        project = "trajectory-anomalies-ubn"
    else:
        project = "trajectory-anomalies-hrubn"
        
    
    if args['wandb']:
        wandb.init(
            settings=wandb.Settings(start_method="fork"),
            project=project,
            config={
            "lr": args['lr'],
            "arch": args['model'],
            "loss": args['loss'],
            "pred_length": args['pred_length'],
            "rec_length": args['rec_length'],
            "epochs": args['epochs'],
            "batch_size": args['batch_size'],
            "weight_decay": args['weight_decay'],
            "lambda1": args['lambda1'],
            "lambda2": args['lambda2'],
            "lambda3": args['lambda3'],
            }
        )
        if 'custom' in args['model']:
            wandb.log({"embed_dim":args['embed_dim'], 
                      "depth":args['depth'], 
                      "num_heads":args['num_heads'], 
                      "decoder_embed_dim":args['decoder_embed_dim'], 
                      "decoder_depth":args['decoder_depth'], 
                      "decoder_num_heads":args['decoder_num_heads']})

    try:
        os.makedirs(args['weights'])
    except OSError:
        print(f' \n directory for the weights already exists. WEIGHTS WILL BE OVERWRITTEN!!! \n')
        pass

    device = torch.device(args['gpu_id'] if args['gpu_id'] != -1 else "cpu")

    local_input_dim,dataset_train,dataset_val,bb_scaler,joint_scaler,out_scaler=create_train_val_datasets(args)
    global_input_dim = 4

    print(f'Num of skeletons sequences: {len(dataset_train)} train, {len(dataset_val)} val')

    res = np.array([int(dim) for dim in args['video_resolution'].split('x')], dtype=np.float32)
    data_test = []
    for camera_id in sorted(os.listdir(os.path.join(args['testdata'], 'trajectories'))):
        tpath = os.path.join(os.path.join(args['testdata'], 'trajectories'), camera_id)
        masks = load_anomaly_masks(os.path.join(args['testdata'], 'frame_level_masks', camera_id))
        ids, frames, X_bb, X_joints, X_out, _, _, _ = load_evaluation_data(bb_scaler,joint_scaler,out_scaler,tpath,inp_len=args['input_length'],inp_gap=0,pred_len=args['pred_length'],res=res,bb_norm='zero_one',joint_norm='zero_one',out_norm='zero_one', rec_data=True,sort='avenue' in args['testdata'].lower())
        data_test.append((masks, ids, frames, X_bb, X_joints, X_out))
    

    train_loader = torch.utils.data.DataLoader(dataset_train, shuffle=True, batch_size=args['batch_size'], num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(dataset_val, shuffle=False, batch_size=args['batch_size'], num_workers=4, pin_memory=True)


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
    model = model.to(device)
    print(f"num of parameters - {sum([m.numel() for m in model.parameters()])}")
    if args['parallel']:
        model = nn.DataParallel(model)

    if args['chkp']:
        model.load_state_dict(torch.load(args['chkp'], map_location=device)["model"])
        print(f"Loaded pretrained weights for the model")

    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    scheduler = optim.lr_scheduler. MultiStepLR(optimizer, milestones=[100], gamma=0.5)

    logname = 'logs/' + datetime.datetime.now().strftime('%Y%m%d_%Hh%M') if args['logname'] is None else args['logname']
    scaler = torch.cuda.amp.GradScaler()
    bformat='{l_bar}{bar}| {n_fmt}/{total_fmt} {rate_fmt}{postfix}'
    
    max_AUC = {'past':0., 'present':0., 'future':0.}
    
    
    for epoch in range(args['epochs']):
        
        if args['eval_only'] and epoch>0:
            break
        
        print("Epoch: %02d"%epoch)
        stats = {}
        if args['eval_only']:
            phases = ['val,past', 'val,present', 'val,future']
        else:
            phases = ['train', 'val,past', 'val,present', 'val,future']
        for phase in phases:
            loss_meter = utils.AverageMeter()
            
            if phase == 'train':
                pbar = tqdm(enumerate(train_loader), total=len(train_loader), bar_format=bformat, ascii='░▒█')
            else:
                pbar = tqdm(enumerate(val_loader), total=len(val_loader), bar_format=bformat, ascii='░▒█')
                
            with torch.set_grad_enabled(phase == 'train'):
                
                for iteration, (data) in pbar:
                    if phase=='train':
                        setting = phase
                    else:
                        setting = phase.split(',')[-1]
            
                    data_skeleton = [d.to(device, non_blocking=True) for d in data]

                    inputs_sk, target_sk = data_skeleton[:3], data_skeleton[6:]
                    inputs_sk = [torch.cat((inputs_sk[0],target_sk[0]),dim=1), torch.cat((inputs_sk[1],target_sk[1]),dim=1), torch.cat((inputs_sk[2],target_sk[2]),dim=1)]

                    losses,eloss,output,target_sk = model(inputs_sk,setting,compute_loss=True)
                    if phase=='train':
                        loss = sum(losses[:-1]) + eloss
                    else:
                        loss = losses[-1]
                    
                    if not math.isfinite(loss):
                        print("Loss is {}, stopping training".format(loss.item()))
                        if args['wandb']:
                            wandb.finish()
                        return [v[0] for v in stats.values()], [v[1] for v in stats.values()]
            
                    loss_meter.update(loss.item(), inputs_sk[0].shape[0])
                    pbar.set_description(f"[{epoch + 1}/{args['epochs']}]")
                    pbar.set_postfix_str(f"[{loss_meter.avg:.2e}|{loss.item():.2e}]")
                    pbar.update()  

                    if phase == 'train':
                        if args['wandb']:
                            wandb.log({"train_loss_per_step": loss,
                                        "train_global_loss_per_step": losses[0],
                                        "train_local_loss_per_step": losses[1],
                                        "train_out_loss_per_step": losses[2],
                                        "lr_per_step": optimizer.param_groups[0]["lr"], 
                                    })
                        optimizer.zero_grad()
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        #scheduler.step()
                    elif args['wandb']:
                        wandb.log({f"val_{setting}_loss_per_step": loss,
                                    f"val_{setting}_global_loss_per_step": losses[0],
                                    f"val_{setting}_local_loss_per_step": losses[1],
                                    f"val_{setting}_out_loss_per_step": losses[2], 
                                    })
                    
                    # cleanup GPU RAM
                    del inputs_sk, target_sk, output, loss
                    
                if 'val' in phase:
                    
                    auc_pred, _, _ = prediction_auc_score(model, data_test, reconstruct_original_data=True,
                                                batch_size=args['batch_size'], setting=setting, is_avenue='avenue' in args['trajectories'].lower())
                    print(f'Test setting {setting}: [MSE: {loss_meter.avg:.6f} | AUC: {auc_pred:.4f}]')
                    stats[setting] = [loss_meter.avg,auc_pred]
                    if args['wandb']:
                        wandb.log({"epoch": epoch,
                                    f"val_{setting}_loss": loss_meter.avg,
                                    f"val_{setting}_AUC": auc_pred
                                    })
                elif args['wandb']:
                    wandb.log({"epoch": epoch,
                            "train_loss": loss_meter.avg,
                            "lr": optimizer.param_groups[0]["lr"],
                            })
        
        sum_mae = sum([v[0] for v in stats.values()])/len(stats)
        sum_auc = sum([v[1] for v in stats.values()])/len(stats) 
        
        if sum_auc > sum([v for v in max_AUC.values()])/len(max_AUC): 
            for s in stats.keys():
                max_AUC[s] = stats[s][1]
                if args['save_best']:
                    state = {'model_name': args['model'], 'model': model.state_dict(), 'epoch': epoch,
                    'input_length': model.input_length, 'prediction_length': model.prediction_length,
                    'bb_scaler': bb_scaler, 'joint_scaler': joint_scaler, 'out_scaler': out_scaler}
                    torch.save(state, 'best_ckpt.pt')
        print(f"AVG : [MSE: {sum_mae:.6f} | AUC: {sum_auc:.4f}]")
        if args['wandb']:
            wandb.log({f"max_AUC_{setting}": max_AUC[setting] for setting in max_AUC.keys()})
            wandb.log({"epoch": epoch,
                    "val_avg_loss": sum_mae,
                    "val_avg_AUC": sum_auc
                    })        
        scheduler.step()
        state = {'model_name': args['model'], 'model': model.state_dict(), 'epoch': epoch,
                'input_length': model.input_length, 'prediction_length': model.prediction_length,
                'bb_scaler': bb_scaler, 'joint_scaler': joint_scaler, 'out_scaler': out_scaler}
        if not os.path.isdir(logname):
            os.makedirs(logname)
        torch.save(state, f'{logname}/ckpt{epoch}.pt')
    if args['wandb']:
        wandb.log({f"max_AUC_{setting}": max_AUC[setting] for setting in max_AUC.keys()}) 
        wandb.finish()
    return max_AUC


if __name__=='__main__':
    model_choices=['trajrec_tiny','trajrec_small','trajrec_base','trajrec_large','trajrec_huge','trajrec_custom']
    parser = argparse.ArgumentParser(description='Skeleton based anomaly detection.')
    parser.add_argument('--seed', default=0, type=int, help='Randomness seed for reproducible training')
    parser.add_argument('--gpu_id', default=0, type=int, help='Which GPUs to use. -1 for cpu')
    parser.add_argument('--parallel', default=False, type=lambda x: (str(x).lower() == 'true'), help='Perform dataparallel training.')
    parser.add_argument('--trajectories', type=str, required=True,
                        help='Path to directory containing training trajectories. For each video in the '
                                'training set, there must be a folder inside this directory containing the '
                                'trajectories.')
    parser.add_argument('--testdata', type=str, required=True,
                        help='Path to directory containing test trajectories and anomaly masks.')
    parser.add_argument('--video_resolution', default='856x480', type=str,
                        help='Resolution of the trajectories\' original video(s). It should be specified '
                                'as WxH, where W is the width and H the height of the video.')
    parser.add_argument('--model', default='trajrec_tiny', choices=model_choices,
                        help='Model architecture to use')
    parser.add_argument('--embed_dim', default=64, type=int, help='Embedding dimension (encoder)')
    parser.add_argument('--depth', default=4, type=int, help='Number of layers (encoder)')
    parser.add_argument('--num_heads', default=4, type=int, help='Number of attention heads (encoder)')
    parser.add_argument('--decoder_embed_dim', default=64, type=int, help='Embedding dimension (decoder)')
    parser.add_argument('--decoder_depth', default=4, type=int, help='Number of layers (decoder)')
    parser.add_argument('--decoder_num_heads', default=4, type=int, help='Number of attention heads (decoder)')
    parser.add_argument('--cross_layers', default=(1, 3), nargs='*', type=int,
                        help='Specify which layers must use cross view attention')
    parser.add_argument('--fusion', choices=('concat_output', 'concat_features', 'encoder'),
                        help='Fusion strategy to use in multiview model')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Learning rate of the optimiser.')
    parser.add_argument('--loss', default='mse', type=str, choices=['log_loss', 'mae', 'mse'],
                        help='Loss function to be minimised by the optimiser.')
    parser.add_argument('--epochs', default=70, type=int, help='Maximum number of epochs for training.')
    parser.add_argument('--batch_size', default=512, type=int, help='Mini-batch size for model training.')
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--lambda1', default=3.0, type=float)
    parser.add_argument('--lambda2', default=3.0, type=float)
    parser.add_argument('--lambda3', default=5.0, type=float)
    parser.add_argument('--input_length', default=12, type=int,
                                help='Number of input time-steps to encode.')
    parser.add_argument('--reconstruct_reverse',type=lambda x: (str(x).lower() == 'true'), default=True,
                                help='Whether to reconstruct the reverse of the input sequence or not.')
    parser.add_argument('--pred_length', default=6, type=int,
                                help='Number of time-steps to predict into future. Ignored if 0.')

    parser.add_argument('--rec_length', default=12, type=int,
                                        help='Number of time-steps to decode from the input sequence.')
    parser.add_argument('--weights', type=str, default='weights', help='Path to directory of the model weights.')
    parser.add_argument('--logname', default=None,
                        help='Name for the log directory and saved model (default: current time)')
    parser.add_argument('--chkp', type=str, help='Path to the checkpoint for loading the pretrained weights')
    parser.add_argument('--setting', type=str, default='future', help='Setting name')
    parser.add_argument('--wandb',default=True,type=lambda x: (str(x).lower() == 'true'),help='Bool indicating if to use wandb')
    parser.add_argument('--save_best',default=True,type=lambda x: (str(x).lower() == 'true'),help='Bool if to save the checkpoint with best (avg) AUC')
    parser.add_argument('--eval_only',default=False,type=lambda x: (str(x).lower() == 'true'),help='Bool if to only run inference.')

    _args = parser.parse_args()
    _args = vars(_args)

    aucs = run(_args)
    print(aucs)


