from copy import deepcopy
import os,sys

import joblib
import numpy as np

from trajectories import load_trajectories, remove_short_trajectories, input_trajectories_missing_steps, extract_global_features, scale_trajectories, change_coordinate_system
from utils import memory


def aggregate_autoencoder_data(trajectories):
    """Put all trajectories into a single big numpy array."""
    X = []
    for trajectory in trajectories.values():
        X.append(trajectory.coordinates)

    return np.vstack(X)


def split_into_train_and_test(trajectories, train_ratio=0.8, seed=42):
    np.random.seed(seed)

    trajectories_ids = []
    trajectories_lengths = []
    for trajectory_id, trajectory in sorted(trajectories.items()):
        trajectories_ids.append(trajectory_id)
        trajectories_lengths.append(len(trajectory))

    sorting_indices = np.argsort(trajectories_lengths)
    q1_idx = round(len(sorting_indices) * 0.25)
    q2_idx = round(len(sorting_indices) * 0.50)
    q3_idx = round(len(sorting_indices) * 0.75)

    sorted_ids = np.array(trajectories_ids)[sorting_indices]
    train_ids = []
    val_ids = []
    quantiles_indices = [0, q1_idx, q2_idx, q3_idx, len(sorting_indices)]
    for idx, q_idx in enumerate(quantiles_indices[1:], 1):
        q_ids = sorted_ids[quantiles_indices[idx - 1]:q_idx]
        q_ids = np.random.permutation(q_ids)
        train_idx = round(len(q_ids) * train_ratio)
        train_ids.extend(q_ids[:train_idx])
        val_ids.extend(q_ids[train_idx:])

    trajectories_train = {}
    for train_id in train_ids:
        trajectories_train[train_id] = trajectories[train_id]

    trajectories_val = {}
    for val_id in val_ids:
        trajectories_val[val_id] = trajectories[val_id]

    return trajectories_train, trajectories_val


def aggregate_rnn_autoencoder_data(trajectories, input_length, input_gap=0, pred_length=0):
    """
    Split each skeleton trajectory into smaller (overlapping) fixed size segments and put them all in a single large
    numpy array.
    """
    Xs, Xs_pred = [], []
    for trajectory in trajectories.values():
        X, X_pred = _aggregate_rnn_autoencoder_data(trajectory.coordinates, input_length, input_gap, pred_length)
        Xs.append(X)
        if X_pred is not None:
            Xs_pred.append(X_pred)

    Xs = np.vstack(Xs)
    if not Xs_pred:
        Xs_pred = None
    else:
        Xs_pred = np.vstack(Xs_pred)

    return Xs, Xs_pred


def _aggregate_rnn_autoencoder_data(coordinates, input_length, input_gap=0, pred_length=0):
    """
    Split a skeleton trajectory into an array smaller (overlapping) fixed size segments.
    """
    input_trajectories, future_trajectories = [], None
    total_input_seq_len = input_length + input_gap * (input_length - 1)
    step = input_gap + 1
    if pred_length > 0:
        future_trajectories = []
        stop = len(coordinates) - pred_length - total_input_seq_len + 1
        for start_index in range(0, stop):
            stop_index = start_index + total_input_seq_len
            input_trajectories.append(coordinates[start_index:stop_index:step, :])
            future_trajectories.append(coordinates[stop_index:(stop_index + pred_length), :])
        input_trajectories = np.stack(input_trajectories, axis=0)
        future_trajectories = np.stack(future_trajectories, axis=0)
    else:
        stop = len(coordinates) - total_input_seq_len + 1
        for start_index in range(0, stop):
            stop_index = start_index + total_input_seq_len
            input_trajectories.append(coordinates[start_index:stop_index:step, :])
        input_trajectories = np.stack(input_trajectories, axis=0)

    return input_trajectories, future_trajectories


@memory.cache
def create_train_val_v2(trajectories_path, video_resolution, input_length, pred_length, reconstruct_original_data=True,
                        input_missing_steps=False, global_normalisation_strategy='zero_one',
                        local_normalisation_strategy='zero_one', out_normalisation_strategy='zero_one'):
    video_resolution = [float(measurement) for measurement in video_resolution.split('x')]
    video_resolution = np.array(video_resolution, dtype=np.float32)

    trajectories = load_trajectories(trajectories_path)
    #print('\nLoaded %d trajectories.' % len(trajectories))


    trajectories = remove_short_trajectories(trajectories, input_length=input_length,
                                             input_gap=0, pred_length=pred_length)
    #print('\nRemoved short trajectories. Number of trajectories left: %d.' % len(trajectories))

    trajectories_train, trajectories_val = split_into_train_and_test(trajectories, train_ratio=0.98, seed=42)

    if input_missing_steps:
        trajectories_train = input_trajectories_missing_steps(trajectories_train)
    #    print('\nInputted missing steps of trajectories.')

    # TODO: General function to extract features
    # X_..._train, X_..._val, y_..._train, y_..._val, ..._scaler = general_function()

    # Global
    global_trajectories_train = extract_global_features(deepcopy(trajectories_train), video_resolution=video_resolution)
    global_trajectories_val = extract_global_features(deepcopy(trajectories_val), video_resolution=video_resolution)

    global_trajectories_train = change_coordinate_system(global_trajectories_train, video_resolution=video_resolution,
                                                         coordinate_system='global', invert=False)
    global_trajectories_val = change_coordinate_system(global_trajectories_val, video_resolution=video_resolution,
                                                       coordinate_system='global', invert=False)
    #print('\nChanged global trajectories\'s coordinate system to global.')

    _, global_scaler = scale_trajectories(aggregate_autoencoder_data(global_trajectories_train),
                                          strategy=global_normalisation_strategy)

    X_global_train, y_global_train = aggregate_rnn_autoencoder_data(global_trajectories_train,
                                                                    input_length=input_length,
                                                                    input_gap=0, pred_length=pred_length)
    X_global_val, y_global_val = aggregate_rnn_autoencoder_data(global_trajectories_val, input_length=input_length,
                                                                input_gap=0, pred_length=pred_length)

    X_global_train, _ = scale_trajectories(X_global_train, scaler=global_scaler, strategy=global_normalisation_strategy)
    X_global_val, _ = scale_trajectories(X_global_val, scaler=global_scaler, strategy=global_normalisation_strategy)
    if y_global_train is not None and y_global_val is not None:
        y_global_train, _ = scale_trajectories(y_global_train, scaler=global_scaler,
                                               strategy=global_normalisation_strategy)
        y_global_val, _ = scale_trajectories(y_global_val, scaler=global_scaler, strategy=global_normalisation_strategy)
    #print('\nNormalised global trajectories using the %s normalisation strategy.' % global_normalisation_strategy)

    # Local
    local_trajectories_train = deepcopy(trajectories_train) if reconstruct_original_data else trajectories_train
    local_trajectories_val = deepcopy(trajectories_val) if reconstruct_original_data else trajectories_val

    local_trajectories_train = change_coordinate_system(local_trajectories_train, video_resolution=video_resolution,
                                                        coordinate_system='bounding_box_centre', invert=False)
    local_trajectories_val = change_coordinate_system(local_trajectories_val, video_resolution=video_resolution,
                                                      coordinate_system='bounding_box_centre', invert=False)
    #print('\nChanged local trajectories\'s coordinate system to bounding_box_centre.')

    _, local_scaler = scale_trajectories(aggregate_autoencoder_data(local_trajectories_train),
                                         strategy=local_normalisation_strategy)

    X_local_train, y_local_train = aggregate_rnn_autoencoder_data(local_trajectories_train, input_length=input_length,
                                                                  input_gap=0, pred_length=pred_length)
    X_local_val, y_local_val = aggregate_rnn_autoencoder_data(local_trajectories_val, input_length=input_length,
                                                              input_gap=0, pred_length=pred_length)

    X_local_train, _ = scale_trajectories(X_local_train, scaler=local_scaler, strategy=local_normalisation_strategy)
    X_local_val, _ = scale_trajectories(X_local_val, scaler=local_scaler, strategy=local_normalisation_strategy)
    if y_local_train is not None and y_local_val is not None:
        y_local_train, _ = scale_trajectories(y_local_train, scaler=local_scaler, strategy=local_normalisation_strategy)
        y_local_val, _ = scale_trajectories(y_local_val, scaler=local_scaler, strategy=local_normalisation_strategy)
    #print('\nNormalised local trajectories using the %s normalisation strategy.' % local_normalisation_strategy)

    # (Optional) Reconstruct the original data
    if reconstruct_original_data:
        #print('\nReconstruction/Prediction target is the original data.')
        out_trajectories_train = trajectories_train
        out_trajectories_val = trajectories_val

        out_trajectories_train = change_coordinate_system(out_trajectories_train, video_resolution=video_resolution,
                                                          coordinate_system='global', invert=False)
        out_trajectories_val = change_coordinate_system(out_trajectories_val, video_resolution=video_resolution,
                                                        coordinate_system='global', invert=False)
        #print('\nChanged target trajectories\'s coordinate system to global.')

        _, out_scaler = scale_trajectories(aggregate_autoencoder_data(out_trajectories_train),
                                           strategy=out_normalisation_strategy)

        X_out_train, y_out_train = aggregate_rnn_autoencoder_data(out_trajectories_train, input_length=input_length,
                                                                  input_gap=0, pred_length=pred_length)
        X_out_val, y_out_val = aggregate_rnn_autoencoder_data(out_trajectories_val, input_length=input_length,
                                                              input_gap=0, pred_length=pred_length)

        X_out_train, _ = scale_trajectories(X_out_train, scaler=out_scaler, strategy=out_normalisation_strategy)
        X_out_val, _ = scale_trajectories(X_out_val, scaler=out_scaler, strategy=out_normalisation_strategy)
        if y_out_train is not None and y_out_val is not None:
            y_out_train, _ = scale_trajectories(y_out_train, scaler=out_scaler, strategy=out_normalisation_strategy)
            y_out_val, _ = scale_trajectories(y_out_val, scaler=out_scaler, strategy=out_normalisation_strategy)
        #print('\nNormalised target trajectories using the %s normalisation strategy.' % out_normalisation_strategy)
    else:
        out_scaler = None

    if y_global_train is not None:
        if reconstruct_original_data:
            # X_global_train, X_local_train, X_out_train, y_global_train, y_local_train, y_out_train = \
            #     shuffle(X_global_train, X_local_train, X_out_train,
            #             y_global_train, y_local_train, y_out_train, random_state=42)
            X_train = [X_global_train, X_local_train, X_out_train]
            y_train = [y_global_train, y_local_train, y_out_train]
            val_data = ([X_global_val, X_local_val, X_out_val], [y_global_val, y_local_val, y_out_val])
        else:
            # X_global_train, X_local_train, y_global_train, y_local_train = \
            #     shuffle(X_global_train, X_local_train, y_global_train, y_local_train, random_state=42)
            X_train = [X_global_train, X_local_train]
            y_train = [y_global_train, y_local_train]
            val_data = ([X_global_val, X_local_val], [y_global_val, y_local_val])
    else:
        if reconstruct_original_data:
            # X_global_train, X_local_train, X_out_train = \
            #     shuffle(X_global_train, X_local_train, X_out_train, random_state=42)
            X_train = [X_global_train, X_local_train, X_out_train]
            y_train = None
            val_data = ([X_global_val, X_local_val, X_out_val],)
        else:
            # X_global_train, X_local_train = shuffle(X_global_train, X_local_train, random_state=42)
            X_train = [X_global_train, X_local_train]
            y_train = None
            val_data = ([X_global_val, X_local_val],)

    return X_train, y_train, val_data, trajectories_train, trajectories_val, global_scaler, local_scaler, out_scaler


def _construct_output_data_alt(multiple_outputs, reconstruction_length, reconstruct_reverse, prediction_length, X_out,
                               y_out=None, X_global=None, y_global=None, X_local=None, y_local=None):
    """
    Put (some of) X_global, X_local, ..., y_out into a list, optionally reversing the X arrays in time (for the RNN
    reconstruction).
    """
    y = []

    if multiple_outputs:
        if reconstruct_reverse:
            y.append(X_global[:, (reconstruction_length - 1)::-1, :])
            y.append(X_local[:, (reconstruction_length - 1)::-1, :])
            y.append(X_out[:, (reconstruction_length - 1)::-1, :])
        else:
            y.append(X_global[:, :reconstruction_length, :])
            y.append(X_local[:, :reconstruction_length, :])
            y.append(X_out[:, :reconstruction_length, :])

        if prediction_length > 0:
            y.append(y_global)
            y.append(y_local)
            y.append(y_out)
    else:
        if reconstruct_reverse:
            y.append(X_out[:, (reconstruction_length - 1)::-1, :])
        else:
            y.append(X_out[:, :reconstruction_length, :])

        if prediction_length > 0:
            y.append(y_out)

    return y


def aggregate_rnn_ae_evaluation_data(trajectories, input_length):
    trajectories_ids, frames, X = [], [], []
    for trajectory in trajectories.values():
        traj_ids, traj_frames, traj_X = _aggregate_rnn_ae_evaluation_data(trajectory, input_length)
        trajectories_ids.append(traj_ids)
        frames.append(traj_frames)
        X.append(traj_X)

    trajectories_ids, frames, X = np.vstack(trajectories_ids), np.vstack(frames), np.vstack(X)

    return trajectories_ids, frames, X


def _aggregate_rnn_ae_evaluation_data(trajectory, input_length):
    traj_frames, traj_X = [], []
    coordinates = trajectory.coordinates
    frames = trajectory.frames

    total_input_seq_len = input_length
    stop = len(coordinates) - total_input_seq_len + 1
    for start_index in range(stop):
        stop_index = start_index + total_input_seq_len
        traj_X.append(coordinates[start_index:stop_index, :])
        traj_frames.append(frames[start_index:stop_index])

    traj_frames, traj_X = np.stack(traj_frames, axis=0), np.stack(traj_X, axis=0)

    trajectory_id = trajectory.trajectory_id
    traj_ids = np.full(traj_frames.shape, fill_value=trajectory_id)

    return traj_ids, traj_frames, traj_X


def load_scalers(pretrained_model_path):
    model_files = os.listdir(pretrained_model_path)
    global_scaler_file = model_files[model_files.index('global_scaler.pkl')]
    local_scaler_file = model_files[model_files.index('local_scaler.pkl')]
    try:
        out_scaler_file = model_files[model_files.index('out_scaler.pkl')]
    except ValueError:
        out_scaler_file = None

    global_scaler = joblib.load(filename=os.path.join(pretrained_model_path, global_scaler_file))
    local_scaler = joblib.load(filename=os.path.join(pretrained_model_path, local_scaler_file))
    if out_scaler_file is not None:
        out_scaler = joblib.load(filename=os.path.join(pretrained_model_path, out_scaler_file))
    else:
        out_scaler = None

    return global_scaler, local_scaler, out_scaler


def load_evaluation_data(global_scaler, 
                         local_scaler, out_scaler, 
                         trajectories_path, 
                         inp_len=12,
                         inp_gap=0,
                         pred_len=6,
                         res=[856,480],
                         bb_norm='zero_one',
                         joint_norm='zero_one',
                         out_norm='zero_one', 
                         rec_data=True,
                         sort=False):
    trajectories = load_trajectories(trajectories_path, sort)

    trajectories = remove_short_trajectories(trajectories, input_length=inp_len,
                                             input_gap=inp_gap, pred_length=pred_len)

    global_trajectories = extract_global_features(deepcopy(trajectories), video_resolution=res)
    global_trajectories = change_coordinate_system(global_trajectories, video_resolution=res,
                                                   coordinate_system='global', invert=False)
    
    trajectories_ids, frames, X_global = aggregate_rnn_ae_evaluation_data(global_trajectories,
                                                                          input_length=inp_len+pred_len)
    X_global, _ = scale_trajectories(X_global, scaler=global_scaler, strategy=bb_norm)

    local_trajectories = deepcopy(trajectories)
    local_trajectories = change_coordinate_system(local_trajectories, video_resolution=res,
                                                  coordinate_system='bounding_box_centre', invert=False)
    _, _, X_local = aggregate_rnn_ae_evaluation_data(local_trajectories, input_length=inp_len+pred_len)
    X_local, _ = scale_trajectories(X_local, scaler=local_scaler, strategy=joint_norm)

    original_trajectories = deepcopy(trajectories)
    _, _, X_original = aggregate_rnn_ae_evaluation_data(original_trajectories, input_length=inp_len+pred_len)

    if rec_data:
        out_trajectories = trajectories
        out_trajectories = change_coordinate_system(out_trajectories, video_resolution=res,
                                                    coordinate_system='global', invert=False)
        _, _, X_out = aggregate_rnn_ae_evaluation_data(out_trajectories, input_length=inp_len+pred_len)
        X_out, _ = scale_trajectories(X_out, scaler=out_scaler, strategy=out_norm)
    else:
        X_out = None

    
    return trajectories_ids, frames, X_global, X_local, X_out, global_scaler, local_scaler, out_scaler
