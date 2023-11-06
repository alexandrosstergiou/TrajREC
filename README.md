# TrajREC

![supported versions](https://img.shields.io/badge/python->3.10-brightgreen/?style=flat&logo=python&color=green)
![Library](https://img.shields.io/badge/library-PyTorch-blue?logo=Pytorch)
![GitHub license](https://img.shields.io/cocoapods/l/AFNetworking)

Code for multitask trajectory anomaly detection

<i></i>
<br>
<i><p align="center"> To appear in <a href="https://wacv2024.thecvf.com"> IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2024</a></p></i>
<p align="center">
<a href="https://arxiv.org/abs/2311.01851" target="blank" >[arXiv preprint]</a>
&nbsp;&nbsp;&nbsp;
<a href="https://alexandrosstergiou.github.io/project_pages/TrajREC/index.html" target="_blank">[project website]</a>
&nbsp;&nbsp;&nbsp;
</p>

<p align="center">
<img src="./images/TrajREC.png" width="700" />
</p>

## Abstract
Video anomaly detection deals with the recognition of abnormal events in videos. Apart from the visual signal, video anomaly detection has also been addressed with the use of skeleton sequences. We propose a holistic representation of skeleton trajectories to learn expected motions across segments at different times. Our approach uses multitask learning to reconstruct any continuous unobserved temporal segment of the trajectory allowing the extrapolation of past or future segments and the interpolation of in-between segments. We use an end-to-end attention-based encoder-decoder. We encode temporally occluded trajectories, jointly learn latent representations of the occluded segments, and reconstruct trajectories based on expected motions across different temporal segments. Extensive experiments on three trajectory-based video anomaly detection datasets show the advantages and effectiveness of our approach with state-of-the-art results on anomaly detection in skeleton trajectories. <p align="center">




## Dependencies

The required packages are listed below

- `einops >= 0.6.0`
- `huggingface-hub >= 0.18.0`
- `numpy >= 1.22.4`
- `pandas >= 1.3.4`
- `tqdm >= 4.65.0`
- `timm >= 0.9.8`
- `torch >= 1.13.0`
- `torchvision >= 0.14.0`
- `scikit-learn >= 1.3.2`
- `scipy >= 1.11.3`
- `wandb >= 0.15.12`

It is recommended to create and use the environment from the provided YAML file:

```
$ conda env create -p ./envs -f env.yml
$ conda activate trajrec
```

> ! Disclaimer: The repository's codebase is structured on Romero Barata's [repo](https://github.com/RomeroBarata/skeleton_based_anomaly_detection)

## Datasets

The original skeleton trajectories at frames 1 and 2 is a duplicates, causing an offset of 1 for subsequent trajectories.
This can be corrected using the script provided:

```
$ python fix_skeleton_data.py --datadir path_to_HR-ShanghaiTech/training/trajectories
```

This will create a new directory `trajectories-corrected` containing the corrected trajectories.

You can download all the datasets from [here](https://drive.google.com/file/d/1TSqZgE6_DH_abnAx2iIc4FHfZxgyUFmz/view?usp=sharing)

The following directory structure is expected:

```
├── data
│   ├── HR-ShanghaiTech
│   │   ├── training
|   |   |   ├── trajectories
|   |   |   |   ├── 00/<traj_id>_<vid_id>/*.csv
|   |   |   ├── trajectories_corrected
|   |   |   |   ├── 00/<traj_id>_<vid_id>/*.csv
│   │   ├── testing
|   |   |   ├── frame_level_masks/<video_id>/*.npy
|   |   |   ├── trajectories
|   |   |   |   ├── <traj_id>/<traj_id>_<vid_id>/*.csv
...

```

## Usage


You can download saved weights for [`HR-STC`](https://drive.google.com/file/d/1hJQAE5VvzxxV_8dGtAI2I2RPcbwpC4Ct/view?usp=sharing) and [`HR-Avenue`](https://drive.google.com/file/d/1TYsXqn91jLzvLGdqozm5jxaVXlyVKgz4/view?usp=sharing)

For training and testing you can call to `run.py`. e.g. with default parameters:

```
$ python run.py --trajectories data/HR-ShanghaiTech/training/trajectories-corrected/00 --testdata data/HR-ShanghaiTech/testing --batch_size 512 --wandb False
```

Use the `help` argument to get a full overview of all available arguments for runs.

For visualising trajectories you will first need to call `generate_reconstructions.py` and then `visualize_skeleton_bbox.py` :

```
$ python generate_reconstructions.py --trajectories data/HR-ShanghaiTech/testing/ --chkp best_ckpt.pt 

$ python3 visualize_skeleton_bbox.py --frames data/ShanghaiTech/testing/frames/01_0014/ --gt_trajectories data/HR-ShanghaiTech/testing/trajectories_corrected/01/0014/ --trajectories reconstructed/predicted_skeletons_future/0014/ --write_dir visualisations

```


## Citation

```bibtex
@inproceedings{stergiou2024holistic,
    title={Holistic Representation Learning for Multitask Trajectory Anomaly Detection},
    author={Stergiou, Alexandros and De Weerdt , Brent and Deligiannis, Nikos},
    booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    year={2024}}
```


## Licence

MIT
