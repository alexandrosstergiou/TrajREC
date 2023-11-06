import joblib
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

memory = joblib.Memory(os.environ['HOME'] + '/.cache/TrajREC')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def numpy_mse(y_pred, y_true):
    eps = 1e-8
    mask = (y_true != 0.0).astype(np.int8)
    a = (y_pred - y_true) ** 2
    return (a * mask).sum(axis=-1) / (mask.sum(axis=-1) + eps)



@torch.no_grad()
def batch_inference(model, x, batch_size=None, setting='future'):
    if batch_size is None:
        batch_size = len(x[0])
    device = next(model.parameters()).device
    dataset = TensorDataset(*(torch.Tensor(d) for d in x))
    dataloader = DataLoader(dataset, batch_size=batch_size)
    output = None
    targets = None
    for batch in dataloader:
        batch = [d.to(device) for d in batch]
        batch, target = model(batch,setting,foreval=True)
        if output is None:
            output = [[] for _ in range(len(batch))]
        if targets is None:
            targets = [[] for _ in range(len(target))]
        for i, tensor in enumerate(batch):
            output[i].append(tensor.detach().cpu().numpy())
        for j, targ in enumerate(target):
            targets[j].append(targ.detach().cpu().numpy())
    output = [np.concatenate(d) for d in output]
    targets = [np.concatenate(d) for d in targets]
    
    return output, targets


def inverse_scale(X, scaler):
    original_shape = X.shape
    input_dim = original_shape[-1]
    X = X.reshape(-1, input_dim)
    X = scaler.inverse_transform(X)
    X = X.reshape(original_shape)
    return X

def restore_global_coordinate_system(X, video_resolution):
    original_shape = X.shape
    X = X.reshape(-1, 2) * video_resolution
    X = X.reshape(original_shape)

    return X

def restore_original_trajectory(reconstructed_X_global, reconstructed_X_local):
    num_examples, input_length, local_input_dim = reconstructed_X_local.shape
    global_input_dim = reconstructed_X_global.shape[-1]
    reconstructed_X_global = reconstructed_X_global.reshape(-1, global_input_dim)
    reconstructed_X_local = reconstructed_X_local.reshape(-1, local_input_dim)
    reps = local_input_dim // 2
    reconstructed_X_traj = reconstructed_X_local * np.tile(reconstructed_X_global[:, -2:], reps=reps)
    reconstructed_X_traj += np.tile(reconstructed_X_global[:, :2], reps=reps)
    reconstructed_X_traj = reconstructed_X_traj.reshape(num_examples, input_length, local_input_dim)

    return reconstructed_X_traj


def compute_bounding_box(keypoints, video_resolution, return_discrete_values=True):
    width, height = video_resolution
    keypoints_reshaped = keypoints.reshape(-1, 2)
    x, y = keypoints_reshaped[:, 0], keypoints_reshaped[:, 1]
    x, y = x[x != 0.0], y[y != 0.0]
    try:
        left, right, top, bottom = np.min(x), np.max(x), np.min(y), np.max(y)
    except ValueError:
        return 0, 0, 0, 0

    extra_width, extra_height = 0.1 * (right - left + 1), 0.1 * (bottom - top + 1)
    left, right = np.clip(left - extra_width, 0, width - 1), np.clip(right + extra_width, 0, width - 1)
    top, bottom = np.clip(top - extra_height, 0, height - 1), np.clip(bottom + extra_height, 0, height - 1)

    if return_discrete_values:
        return int(round(left)), int(round(right)), int(round(top)), int(round(bottom))
    else:
        return left, right, top, bottom


def summarise_reconstruction(reconstructed_X, frames, trajectory_ids):
    unique_ids = np.unique(trajectory_ids)
    num_examples, input_length, input_dim = reconstructed_X.shape
    reconstructed_X = reconstructed_X.reshape(-1, input_dim)
    frames = frames.reshape(-1)
    trajectory_ids = trajectory_ids.reshape(-1)

    all_trajectory_ids, all_summarised_frames, all_summarised_recs = [], [], []
    for trajectory_id in unique_ids:
        mask = trajectory_ids == trajectory_id

        current_frames = frames[mask]
        
        current_reconstructions = reconstructed_X[mask, :]
        summarised_frames, summarised_recs = summarise_reconstruction_per_frame(current_reconstructions, current_frames)
        all_summarised_frames.append(summarised_frames)
        all_summarised_recs.append(summarised_recs)
        all_trajectory_ids.append([trajectory_id] * len(summarised_frames))

    all_trajectory_ids = np.concatenate(all_trajectory_ids)
    all_summarised_frames = np.concatenate(all_summarised_frames)
    all_summarised_recs = np.vstack(all_summarised_recs)

    return all_trajectory_ids, all_summarised_frames, all_summarised_recs


def summarise_reconstruction_per_frame(recs, frames):
    unique_frames = np.unique(frames)
    unique_recs = np.empty((len(unique_frames), recs.shape[-1]), dtype=np.float32)
    for idx, frame in enumerate(unique_frames):
        mask = frames == frame
        unique_recs[idx, :] = np.mean(recs[mask, :], axis=0, keepdims=True)

    return unique_frames, unique_recs


def reconstruct_data(x, video_resolution, reconstruct_original_data, global_scaler, local_scaler, out_scaler):
    if reconstruct_original_data:
        traj = inverse_scale(x, scaler=out_scaler)
        traj = restore_global_coordinate_system(traj, video_resolution=video_resolution)
    else:
        global_traj = inverse_scale(x[..., :4], scaler=global_scaler)
        local_traj = inverse_scale(x[..., 4:], scaler=local_scaler)
        global_traj = restore_global_coordinate_system(global_traj, video_resolution=video_resolution)
        traj = restore_original_trajectory(global_traj, local_traj)
    return traj

def get_vid_and_traj(rec_ids):
    s_ids = np.core.defchararray.split(rec_ids, sep='_')
    v_ids = np.asarray([ids[0] for ids in s_ids])
    traj_ids = np.asarray([ids[1] for ids in s_ids])
    return v_ids, traj_ids


def write_predicted_masks(pretrained_model_path, num_frames_per_video, anomalous_frames, normal_frames,
                          reconstructed_bounding_boxes, rec_ids, reconstruction_frames, video_resolution):
    
    s_ids = np.core.defchararray.split(rec_ids, sep='_')
    v_ids = np.asarray([ids[0] for ids in s_ids])

    unique_video_ids = np.unique(v_ids)
    w, h = int(video_resolution[0]), int(video_resolution[1])
    anomaly_path = os.path.join(pretrained_model_path, 'predicted_pixel_level_anomaly_masks')
    if not os.path.isdir(anomaly_path):
        os.makedirs(anomaly_path)
    normal_path = os.path.join(pretrained_model_path, 'predicted_pixel_level_normal_masks')
    if not os.path.isdir(normal_path):
        os.makedirs(normal_path)

    for video_id in unique_video_ids:
        num_frames = num_frames_per_video[video_id]
        anomaly_mask = np.zeros((num_frames, h, w), dtype=np.uint8)
        normal_mask = np.zeros((num_frames, h, w), dtype=np.uint8)

        mask = v_ids == video_id
        current_anomalous_frames, current_normal_frames = anomalous_frames[mask], normal_frames[mask]
        current_bounding_boxes, current_frames = reconstructed_bounding_boxes[mask, :], reconstruction_frames[mask]
        for idx, frame in enumerate(current_frames):
            bounding_box = current_bounding_boxes[idx, :]
                 
            if current_anomalous_frames[idx] or current_normal_frames[idx]:
                anomaly = np.ones((bounding_box[3] - bounding_box[2] + 1, bounding_box[1] - bounding_box[0] + 1), dtype=np.uint8)
                
                if current_anomalous_frames[idx]:
                    anomaly_mask[frame][bounding_box[3]:(bounding_box[4] + 1), bounding_box[0]:(bounding_box[1] + 1)] = anomaly
                
                elif current_normal_frames[idx]:
                    normal_mask[frame][bounding_box[3]:(bounding_box[4] + 1), bounding_box[0]:(bounding_box[1] + 1)] = anomaly
            
        np.save(os.path.join(anomaly_path, video_id), arr=anomaly_mask)
        np.save(os.path.join(normal_path, video_id), arr=normal_mask)


def write_reconstructed_trajectories(pretrained_model_path, reconstructed_traj,
                                     rec_ids, reconstruction_frames, trajectory_type='skeleton'):

    s_ids = np.core.defchararray.split(rec_ids, sep='_')
    v_ids = np.asarray([ids[0] for ids in s_ids])
    traj_ids = np.asarray([ids[1] for ids in s_ids])
    
    _v_ids = np.unique(v_ids)

    writing_dir = os.path.join(pretrained_model_path, trajectory_type)
    if not os.path.isdir(writing_dir):
        os.makedirs(writing_dir)

    for v_id in _v_ids:
        video_writing_dir = os.path.join(writing_dir, v_id)
        if not os.path.isdir(video_writing_dir):
            os.makedirs(video_writing_dir)

        mask = v_ids == v_id
        current_skeleton_ids = traj_ids[mask]
        current_frames = reconstruction_frames[mask]
        current_recs = reconstructed_traj[mask, :]

        unique_current_skeleton_ids = np.unique(current_skeleton_ids)
        for skeleton_id in unique_current_skeleton_ids:
            skeleton_writing_file = os.path.join(video_writing_dir, skeleton_id) + '.csv'
            mask = current_skeleton_ids == skeleton_id
            current_skeleton_frames = current_frames[mask].reshape(-1, 1)
            current_skeleton_recs = current_recs[mask, :]
            trajectory = np.hstack((current_skeleton_frames, current_skeleton_recs))
            np.savetxt(skeleton_writing_file, trajectory, fmt='%.4f', delimiter=',')