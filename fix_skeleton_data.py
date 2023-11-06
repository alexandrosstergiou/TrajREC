import argparse
import os.path

import numpy as np


def fix_trajectory(trajectory):
    frame_numbers = trajectory[:, 0].astype(np.int32)
    # remove frame index 1
    if frame_numbers[0] == 0:
        trajectory = np.vstack((trajectory[:1], trajectory[2:]))
        trajectory[0, 0] = 1  # move frame index 0 to the removed index 1
    if frame_numbers[0] == 1:
        trajectory = trajectory[1:]
    # shift all one frame backwards
    trajectory[:, 0] -= 1
    return trajectory


def main(args):
    for root, dirs, files in os.walk(args.datadir):
        if args.outputdir is None:
            args.outputdir = os.path.normpath(args.datadir) + '-corrected'
        for filename in files:
            trajectory_path = os.path.join(root, filename)
            trajectory = np.loadtxt(trajectory_path, delimiter=',', ndmin=2)
            trajectory = fix_trajectory(trajectory)
            output_path = os.path.relpath(trajectory_path, start=args.datadir)
            output_path = os.path.join(args.outputdir, output_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.savetxt(output_path, trajectory, delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Correct the training skeleton trajectories.')

    parser.add_argument('--datadir', type=str, required=True,
                        help='The training directory containing skeleton trajectories')
    parser.add_argument('--outputdir', type=str, default=None,
                        help='Output directory to put the corrected trajectories')

    args = parser.parse_args()
    main(args)
