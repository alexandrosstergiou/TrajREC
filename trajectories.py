import glob
import os
import numpy as np
from sklearn.preprocessing import quantile_transform, MinMaxScaler, RobustScaler

from utils import compute_bounding_box, numpy_mse


class StdScaler:
    def __init__(self, stds=3):
        self.stds = stds
        self.mu = None
        self.sigma = None

    def fit(self, X):
        self.mu = np.nanmean(X, axis=0, keepdims=True)
        self.sigma = np.nanstd(X, axis=0, keepdims=True)

    def transform(self, X):
        reps = [X.shape[0], 1]
        mu = np.tile(self.mu, reps=reps)
        sigma = np.tile(self.sigma, reps=reps)
        X = (X - (mu - self.stds * sigma)) / (2 * self.stds * sigma)

        return X

    def inverse_transform(self, X):
        reps = [X.shape[0], 1]
        mu = np.tile(self.mu, reps=reps)
        sigma = np.tile(self.sigma, reps=reps)
        X = X * (2 * self.stds * sigma) + (mu - self.stds * sigma)

        return X

class Trajectory:
    def __init__(self, trajectory_id, frames, coordinates):
        self.trajectory_id = trajectory_id
        self.person_id = trajectory_id.split('_')[1]
        self.frames = frames
        self.coordinates = coordinates
        self.is_global = False

    def __len__(self):
        return len(self.frames)
    
    def use_global_features(self, video_resolution, extract_delta=False, use_first_step_as_reference=False):
        self.coordinates = self._extract_global_features(video_resolution=video_resolution, extract_delta=extract_delta,
                                                         use_first_step_as_reference=use_first_step_as_reference)
    
    def _extract_global_features(self, video_resolution, extract_delta=False, use_first_step_as_reference=False):
        bounding_boxes = np.apply_along_axis(compute_bounding_box, axis=1, arr=self.coordinates,
                                             video_resolution=video_resolution)
        bbs_measures = np.apply_along_axis(self._extract_bounding_box_measurements, axis=1, arr=bounding_boxes)
        bbs_centre = np.apply_along_axis(self._extract_bounding_box_centre, axis=1, arr=bounding_boxes)
        if extract_delta:
            bbs_delta = np.vstack((np.full((1, 2), fill_value=1e-7), np.diff(bbs_centre, axis=0)))

        if use_first_step_as_reference:
            bbs_centre -= bbs_centre[0]
            # bbs_centre /= np.where(bbs_measures == 0.0, 1.0, bbs_measures)
            bbs_centre[0] += 1e-6

        if extract_delta:
            return np.hstack((bbs_centre, bbs_delta, bbs_measures))

        return np.hstack((bbs_centre, bbs_measures))


    @staticmethod
    def _extract_bounding_box_centre(bb):
        x = (bb[0] + bb[1]) / 2
        y = (bb[2] + bb[3]) / 2

        return np.array([x, y], dtype=np.float32)

    @staticmethod
    def _extract_bounding_box_measurements(bb):
        width = bb[1] - bb[0]
        height = bb[3] - bb[2]

        return np.array([width, height], dtype=np.float32)

    def change_coordinate_system(self, video_resolution, coordinate_system='global', invert=False):
        if invert:
            if coordinate_system == 'global':
                self.coordinates = self._from_global_to_image(self.coordinates, video_resolution=video_resolution)
            else:
                raise ValueError('Unknown coordinate system. Only global is available for inversion.')
        else:
            if coordinate_system == 'global':
                self.coordinates = self._from_image_to_global(self.coordinates, video_resolution=video_resolution)
            elif coordinate_system == 'bounding_box_top_left':
                self.coordinates = self._from_image_to_bounding_box(self.coordinates,
                                                                    video_resolution=video_resolution,
                                                                    location='top_left')
            elif coordinate_system == 'bounding_box_centre':
                self.coordinates = self._from_image_to_bounding_box(self.coordinates,
                                                                    video_resolution=video_resolution,
                                                                    location='centre')
            else:
                raise ValueError('Unknown coordinate system. Please select one of: global, bounding_box_top_left, or '
                                 'bounding_box_centre.')

    @staticmethod
    def _from_global_to_image(coordinates, video_resolution):
        original_shape = coordinates.shape
        coordinates = coordinates.reshape(-1, 2) * video_resolution

        return coordinates.reshape(original_shape)

    @staticmethod
    def _from_image_to_global(coordinates, video_resolution):
        original_shape = coordinates.shape
        coordinates = coordinates.reshape(-1, 2) / video_resolution

        return coordinates.reshape(original_shape)

    @staticmethod
    def _from_image_to_bounding_box(coordinates, video_resolution, location='centre'):
        if location == 'top_left':
            fn = Trajectory._from_image_to_top_left_bounding_box
        elif location == 'centre':
            fn = Trajectory._from_image_to_centre_bounding_box
        else:
            raise ValueError('Unknown location for the bounding box. Please select either top_left or centre.')

        coordinates = fn(coordinates, video_resolution=video_resolution)

        return coordinates

    @staticmethod
    def _from_image_to_top_left_bounding_box(coordinates, video_resolution):
        for idx, kps in enumerate(coordinates):
            if any(kps):
                left, right, top, bottom = compute_bounding_box(kps, video_resolution=video_resolution)
                xs, ys = np.hsplit(kps.reshape(-1, 2), indices_or_sections=2)
                xs, ys = np.where(xs == 0.0, float(left), xs), np.where(ys == 0.0, float(top), ys)
                xs, ys = (xs - left) / (right - left), (ys - top) / (bottom - top)
                kps = np.hstack((xs, ys)).ravel()

            coordinates[idx] = kps

        return coordinates

    @staticmethod
    def _from_image_to_centre_bounding_box(coordinates, video_resolution):
        # TODO: Better implementation
        # coordinates = np.where(coordinates == 0, np.nan, coordinates)
        # bounding_boxes = np.apply_along_axis(compute_bounding_box, axis=1, arr=coordinates,
        #                                      video_resolution=video_resolution)
        # centre_x = (bounding_boxes[:, 0] + bounding_boxes[:, 1]) / 2
        # centre_y = (bounding_boxes[:, 2] + bounding_boxes[:, 3]) / 2
        for idx, kps in enumerate(coordinates):
            if any(kps):
                left, right, top, bottom = compute_bounding_box(kps, video_resolution=video_resolution)
                centre_x, centre_y = (left + right) / 2, (top + bottom) / 2
                xs, ys = np.hsplit(kps.reshape(-1, 2), indices_or_sections=2)
                xs, ys = np.where(xs == 0.0, centre_x, xs) - centre_x, np.where(ys == 0.0, centre_y, ys) - centre_y
                left, right, top, bottom = left - centre_x, right - centre_x, top - centre_y, bottom - centre_y
                width, height = right - left, bottom - top
                xs, ys = xs / width, ys / height
                kps = np.hstack((xs, ys)).ravel()

            coordinates[idx] = kps

        return coordinates

    def is_short(self, input_length, input_gap, pred_length=0):
        min_trajectory_length = input_length + input_gap * (input_length - 1) + pred_length

        return len(self) < min_trajectory_length

    def input_missing_steps(self):
        """Fill missing steps with a weighted average of the closest non-missing steps."""
        trajectory_length, input_dim = self.coordinates.shape
        last_step_non_missing = 0
        consecutive_missing_steps = 0
        while last_step_non_missing < trajectory_length - 1:
            step_is_missing = np.sum(self.coordinates[last_step_non_missing + 1, :] == 0) == input_dim
            while step_is_missing:
                consecutive_missing_steps += 1
                step_is_missing = \
                    np.sum(self.coordinates[last_step_non_missing + 1 + consecutive_missing_steps, :] == 0) == input_dim

            if consecutive_missing_steps:
                start_trajectory = self.coordinates[last_step_non_missing, :]
                end_trajectory = self.coordinates[last_step_non_missing + 1 + consecutive_missing_steps, :]
                for n in range(1, consecutive_missing_steps + 1):
                    a = ((consecutive_missing_steps + 1 - n) / (consecutive_missing_steps + 1)) * start_trajectory
                    b = (n / (consecutive_missing_steps + 1)) * end_trajectory
                    fill_step = a + b
                    fill_step = np.where((start_trajectory == 0) | (end_trajectory == 0), 0, fill_step)
                    self.coordinates[last_step_non_missing + n, :] = fill_step

            last_step_non_missing += consecutive_missing_steps + 1
            consecutive_missing_steps = 0


def load_trajectories(trajectories_path, load_ordered=False):
    trajectories = {}
    csv_files = [f for f in glob.iglob('**/*.csv', root_dir=trajectories_path, recursive=True)]
    if load_ordered:
        csv_files = sorted(csv_files)
    for csv_file_name in csv_files:
        trajectory_file_path = os.path.join(trajectories_path, csv_file_name)
        trajectory = np.loadtxt(trajectory_file_path, dtype=np.float32, delimiter=',', ndmin=2)
        trajectory_frames, trajectory_coordinates = trajectory[:, 0].astype(np.int32), trajectory[:, 1:]
        trajectory_id = os.path.splitext(csv_file_name)[0].replace(os.sep, '_')
        if '_' not in trajectory_id:
            trajectory_id = '_' + trajectory_id
            
        trajectories[trajectory_id] = Trajectory(trajectory_id=trajectory_id,
                                                 frames=trajectory_frames,
                                                 coordinates=trajectory_coordinates)
    return trajectories


# def compute_ae_reconstruction_errors(X, reconstructed_X, loss):
#     loss_fn = {'log_loss': binary_crossentropy, 'mae': mean_absolute_error, 'mse': mean_squared_error}[loss]
#     return loss_fn(X, reconstructed_X)

def remove_short_trajectories(trajectories, input_length, input_gap, pred_length=0):
    filtered_trajectories = {}
    for trajectory_id, trajectory in trajectories.items():
        if not trajectory.is_short(input_length=input_length, input_gap=input_gap, pred_length=pred_length):
            filtered_trajectories[trajectory_id] = trajectory

    return filtered_trajectories


def load_anomaly_masks(anomaly_masks_path):
    file_names = os.listdir(anomaly_masks_path)
    masks = {}
    for file_name in file_names:
        full_id = file_name.split('.')[0]
        file_path = os.path.join(anomaly_masks_path, file_name)
        masks[full_id] = np.load(file_path)

    return masks


def assemble_ground_truth_and_reconstructions(anomaly_masks, trajectory_ids,
                                              reconstruction_frames, reconstruction_errors,
                                              return_video_ids=False, return_grouped_scores=False):
    y_true, y_hat = {}, {}
    for full_id in anomaly_masks.keys():
        _, video_id = full_id.split('_')
        y_true[video_id] = anomaly_masks[full_id].astype(np.int32)
        y_hat[video_id] = np.zeros_like(y_true[video_id], dtype=np.float32)

    unique_ids = np.unique(trajectory_ids)
    for trajectory_id in unique_ids:
        video_id, _ = trajectory_id.split('_')
        indices = trajectory_ids == trajectory_id
        frames = reconstruction_frames[indices]
        y_hat[video_id][frames] = np.maximum(y_hat[video_id][frames], reconstruction_errors[indices])

    y_true_, y_hat_, video_ids = [], [], []
    for video_id in sorted(y_true.keys()):
        y_true_.append(y_true[video_id])
        y_hat_.append(y_hat[video_id])
        video_ids.extend([video_id] * len(y_true_[-1]))

    y_true_, y_hat_ = np.concatenate(y_true_), np.concatenate(y_hat_)

    if return_video_ids:
        return y_true_, y_hat_, video_ids
    if return_grouped_scores:
        return y_true_, y_hat_, y_true, y_hat
    else:
        return y_true_, y_hat_


def quantile_transform_errors(y_hats):
    for camera_id, y_hat in y_hats.items():
        y_hats[camera_id] = quantile_transform(y_hat.reshape(-1, 1)).reshape(-1)

    return y_hats


def input_trajectories_missing_steps(trajectories):
    for trajectory in trajectories.values():
        trajectory.input_missing_steps()

    return trajectories

def extract_global_features(trajectories, video_resolution, extract_delta=False, use_first_step_as_reference=False):
    for trajectory in trajectories.values():
        trajectory.use_global_features(video_resolution=video_resolution, extract_delta=extract_delta,
                                       use_first_step_as_reference=use_first_step_as_reference)

    return trajectories

def scale_trajectories(X, scaler=None, strategy='zero_one'):
    original_shape = X.shape
    input_dim = original_shape[-1]
    X = X.reshape(-1, input_dim)

    if strategy == 'zero_one':
        X_scaled, scaler = scale_trajectories_zero_one(X, scaler=scaler)
    elif strategy == 'three_stds':
        X_scaled, scaler = scale_trajectories_three_stds(X, scaler=scaler)
    elif strategy == 'robust':
        X_scaled, scaler = scale_trajectories_robust(X, scaler=scaler)
    else:
        raise ValueError('Unknown strategy. Please select either zero_one or three_stds.')

    X, X_scaled = X.reshape(original_shape), X_scaled.reshape(original_shape)

    return X_scaled, scaler


def scale_trajectories_zero_one(X, scaler=None):
    if scaler is None:
        X = np.where(X == 0.0, np.nan, X)
        X_min = np.nanmin(X, axis=0, keepdims=True)
        X_min = np.where(np.isnan(X_min), 0.0, X_min)
        X_min = np.tile(X_min, reps=[X.shape[0], 1])

        eps = 1e-3
        X = np.where(np.isnan(X), X_min - eps, X)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X)

    num_examples = X.shape[0]
    X_scaled = np.where(X == 0.0, np.tile(scaler.data_min_, reps=[num_examples, 1]), X)
    X_scaled = scaler.transform(X_scaled)

    return X_scaled, scaler


def scale_trajectories_three_stds(X, scaler=None):
    if scaler is None:
        X = np.where(X == 0.0, np.nan, X)

        scaler = StdScaler(stds=3)
        scaler.fit(X)

    X_scaled = np.where(X == 0.0, np.nan, X)
    X_scaled = scaler.transform(X_scaled)
    X_scaled = np.where(np.isnan(X_scaled), 0.0, X_scaled)

    return X_scaled, scaler


def scale_trajectories_robust(X, scaler=None):
    X_scaled = np.where(X == 0.0, np.nan, X)
    if scaler is None:
        scaler = RobustScaler(quantile_range=(10.0, 90.0))
        scaler.fit(X_scaled)

    X_scaled = scaler.transform(X_scaled)
    X_scaled = np.where(np.isnan(X_scaled), 0.0, X_scaled)

    return X_scaled, scaler


def change_coordinate_system(trajectories, video_resolution, coordinate_system='global', invert=False):
    """
    Change the coordinates of the skeletons to difference reference frames.
    If `invert` is false, the original coordinates are in pixels.
        'global': normalize pixel coordinates to the range [0, 1] using `video_resolution`.
        'bounding_box_centre': local pose, normalized to bounding box.
    """
    for trajectory in trajectories.values():
        trajectory.change_coordinate_system(video_resolution, coordinate_system=coordinate_system, invert=invert)

    return trajectories


def compute_worst_mistakes(y_true, y_hat, video_ids, type='false_positives', top=10):
    # sorting_indices = np.argsort(y_hat)
    frames = generate_array_of_frames(video_ids)
    video_ids = np.array(video_ids)

    if type == 'false_positives':
        true_negatives = y_true == 0
        y_hat_ = y_hat[true_negatives]
        video_ids_ = video_ids[true_negatives]
        frames_ = frames[true_negatives]

        sorting_indices = np.argsort(y_hat_)

        indices = sorting_indices[-top:]
    elif type == 'false_negatives':
        true_positives = y_true == 1
        y_hat_ = y_hat[true_positives]
        video_ids_ = video_ids[true_positives]
        frames_ = frames[true_positives]

        sorting_indices = np.argsort(y_hat_)

        indices = sorting_indices[:top]
    else:
        raise ValueError('Unknown mistake type. Please choose either false_positives or false_negatives.')

    return video_ids_[indices], frames_[indices], y_hat_[indices]


def generate_array_of_frames(x):
    """x is already sorted."""
    _, counts = np.unique(x, return_counts=True)
    result = []
    for count in counts:
        result.append(np.arange(count))

    return np.concatenate(result)


def write_all_worst_mistakes(all_pretrained_models_path, worst_false_positives, worst_false_negatives):
    file_path = os.path.join(all_pretrained_models_path, 'mistakes.txt')
    camera_ids = sorted(worst_false_positives.keys())
    with open(file_path, mode='w') as file:
        for camera_id in camera_ids:
            print('\nCamera ID: %s' % camera_id, file=file)
            print('\nWorst False Positives:', file=file)
            video_ids, frames, scores = worst_false_positives[camera_id]
            for video_id, frame, score in zip(video_ids[::-1], frames[::-1], scores[::-1]):
                print('Video ID: %s\tFrame: %d\tRec. Error: %.4f' % (video_id, frame, score), file=file)

            print('\nWorst False Negatives:', file=file)
            video_ids, frames, scores = worst_false_negatives[camera_id]
            for video_id, frame, score in zip(video_ids, frames, scores):
                print('Video ID: %s\tFrame: %d\tRec. Error: %.4f' % (video_id, frame, score), file=file)

    return None


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
    # X_global is already in image coordinates
    # X_local is in bounding_box_coordinates
    num_examples, input_length, local_input_dim = reconstructed_X_local.shape
    global_input_dim = reconstructed_X_global.shape[-1]
    reconstructed_X_global = reconstructed_X_global.reshape(-1, global_input_dim)
    reconstructed_X_local = reconstructed_X_local.reshape(-1, local_input_dim)
    reps = local_input_dim // 2
    reconstructed_X_traj = reconstructed_X_local * np.tile(reconstructed_X_global[:, -2:], reps=reps)
    reconstructed_X_traj += np.tile(reconstructed_X_global[:, :2], reps=reps)
    reconstructed_X_traj = reconstructed_X_traj.reshape(num_examples, input_length, local_input_dim)

    return reconstructed_X_traj


def write_reconstructed_trajectories(pretrained_model_path, reconstructed_traj,
                                     reconstruction_ids, reconstruction_frames, trajectory_type='skeleton'):

    video_ids, skeleton_ids = extract_video_and_skeleton_ids(reconstruction_ids)
    unique_video_ids = np.unique(video_ids)

    writing_dir = os.path.join(pretrained_model_path, trajectory_type)
    if not os.path.isdir(writing_dir):
        os.makedirs(writing_dir)

    for video_id in unique_video_ids:
        video_writing_dir = os.path.join(writing_dir, video_id)
        if not os.path.isdir(video_writing_dir):
            os.makedirs(video_writing_dir)

        mask = video_ids == video_id
        current_skeleton_ids = skeleton_ids[mask]
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


def extract_video_and_skeleton_ids(reconstruction_ids):
    split_ids = np.core.defchararray.split(reconstruction_ids, sep='_')
    video_ids, skeleton_ids = [], []
    for ids in split_ids:
        video_id, skeleton_id = ids
        video_ids.append(video_id)
        skeleton_ids.append(skeleton_id)

    return np.array(video_ids), np.array(skeleton_ids)


def compute_rnn_ae_reconstruction_errors(X, reconstructed_X, loss):
    num_examples, input_length, input_dim = X.shape
    X = X.reshape(-1, input_dim)
    reconstructed_X = reconstructed_X.reshape(-1, input_dim)

    # loss_fn = {'log_loss': binary_crossentropy, 'mae': mean_absolute_error,
    #            'mse': mean_squared_error, 'balanced_mse': balanced_mean_squared_error,
    #            'balanced_mae': balanced_mean_absolute_error}[loss]
    loss_fn = {'mse': numpy_mse}[loss]
    reconstruction_errors = loss_fn(reconstructed_X, X)

    return reconstruction_errors.reshape(num_examples, input_length)


def summarise_reconstruction_errors(reconstruction_errors, frames, trajectory_ids):
    """
    Simplify skeleton trajectory prediction errors by averaging errors of overlapping predictions.
    The result will still have multiple scores per frame numbers when different trajectories (persons) overlap.
    """
    unique_ids = np.unique(trajectory_ids)
    all_trajectory_ids, all_summarised_frames, all_summarised_errors = [], [], []
    for trajectory_id in unique_ids:
        mask = trajectory_ids == trajectory_id
        current_frames = frames[mask]
        current_errors = reconstruction_errors[mask]
        summarised_frames, summarised_errors = summarise_reconstruction_errors_per_frame(current_errors, current_frames)
        all_summarised_frames.append(summarised_frames)
        all_summarised_errors.append(summarised_errors)
        all_trajectory_ids.append([trajectory_id] * len(summarised_frames))

    all_trajectory_ids = np.concatenate(all_trajectory_ids)
    all_summarised_frames = np.concatenate(all_summarised_frames)
    all_summarised_errors = np.concatenate(all_summarised_errors)

    return all_trajectory_ids, all_summarised_frames, all_summarised_errors


def summarise_reconstruction_errors_per_frame(errors, frames):
    unique_frames = np.unique(frames)
    unique_errors = np.empty(unique_frames.shape, dtype=np.float32)
    for idx, frame in enumerate(unique_frames):
        mask = frames == frame
        unique_errors[idx] = np.mean(errors[mask])

    return unique_frames, unique_errors


def retrieve_future_skeletons(trajectories_ids, X, pred_length):
    """
    Shift and cut the skeleton trajectory segments in X to correspond (in time) with the predicted skeleton segments.
    Pads each trajectory with `pred_length` zeros to match unknown data, which are removed again later, see
    `discard_information_from_padded_frames`
    """
    input_dim = X.shape[-1]
    traj_id_per_example = trajectories_ids[:, 0]
    indices = np.unique(traj_id_per_example, return_index=True)[1]
    unique_ids = [traj_id_per_example[idx] for idx in sorted(indices)]

    y = []
    for unique_id in unique_ids:
        current_ids = unique_id == traj_id_per_example
        current_X = X[current_ids, :, :]
        future_X = current_X[pred_length:, -pred_length:, :]
        padding = np.zeros(shape=(pred_length, pred_length, input_dim), dtype=np.float32)
        future_X = np.concatenate((future_X, padding), axis=0)
        y.append(future_X)

    y = np.vstack(y)

    return y



def discard_information_from_padded_frames(pred_ids, pred_frames, pred_errors, pred_length):
    id_per_example = pred_ids[:, 0]
    indices = np.unique(id_per_example, return_index=True)[1]
    unique_ids = [id_per_example[idx] for idx in sorted(indices)]

    all_ids, all_frames, all_errors = [], [], []
    for unique_id in unique_ids:
        current_ids = unique_id == id_per_example
        actual_ids = pred_ids[current_ids][:-pred_length]
        actual_frames = pred_frames[current_ids][:-pred_length]
        actual_errors = pred_errors[current_ids][:-pred_length]

        all_ids.append(actual_ids)
        all_frames.append(actual_frames)
        all_errors.append(actual_errors)

    all_ids = np.vstack(all_ids)
    all_frames = np.vstack(all_frames)
    all_errors = np.vstack(all_errors)

    return all_ids, all_frames, all_errors


def compute_num_frames_per_video(anomaly_masks):
    num_frames_per_video = {}
    for full_id, anomaly_mask in anomaly_masks.items():
        _, video_id = full_id.split('_')
        num_frames_per_video[video_id] = len(anomaly_mask)

    return num_frames_per_video