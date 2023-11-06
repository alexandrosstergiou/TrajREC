import argparse
import math
import numpy as np
import cv2
import os
import tqdm

# BGR
COLOURS = {(0, 1) : (255, 0, 255), 
           (0, 2) : (255, 0, 255), 
           (1, 3) : (255, 0, 255), 
           (2, 4) : (255, 0, 255),
           (5, 7) : (0, 127, 255), # left arm
           (7, 9) : (0, 255, 255), # left arm
           (6, 8) : (127, 255, 0), # right arm
           (8, 10) : (0, 255, 0), # right arm
           (11, 13) : (127, 225, 0), # left leg
           (13, 15) : (255, 225, 0), # left let
           (12, 14) : (255, 127, 0), # right leg
           (14, 16) : (255, 0, 0), # right leg
           (0, 5) : (192, 127, 192), 
           (0, 6) : (127, 127, 192),
           (5, 6) : (0, 0, 255), # chest
           (5, 11) : (0, 0, 255), # left side
           (6, 12) : (0, 0, 255), # right side
           (11, 12) : (0, 0, 255) # pelvis
           }  # Dark Green



COLOURS_POINTS = {
            0 : (255, 0, 255), 
            1 : (255, 0, 255), 
            2 : (255, 0, 255), 
            3 : (255, 0, 255), 
            4 : (255, 0, 255), 
            5 : (0, 64, 255), 
            7 : (0, 191, 255),
            9 : (0, 255, 255),
            6 : (191, 255, 0),
            8 : (64, 255, 0),
            10 : (0, 255, 0),
            11 : (127,255,127),
            13 : (192, 225, 0),
            15 : (255, 255, 0),
            12 : (255, 127, 64),
            14 : (255, 64, 0),
            16 : (255, 0, 0)
           }  # Dark Green



parser = argparse.ArgumentParser(description='Visualize the predicted skeletons with corresponding bounding boxes.')

parser.add_argument('--frames', type=str, help='Directory containing video frames.')
parser.add_argument('--gt_trajectories', type=str,
                              help='Directory containing the ground-truth trajectories of people in the video.')
parser.add_argument('--draw_gt_skeleton', type=bool,default=True, help='Whether to draw the ground-truth skeletons or not.')
parser.add_argument('--draw_gt_bbox',type=bool,default=False, help='Whether to draw the bounding box of the ground-truth skeletons or not.')
parser.add_argument('--trajectories', type=str,help='Directory containing the reconstructed/predicted trajectories of people in '
                                   'the video.')
parser.add_argument('--draw_pred_skeleton',type=bool,default=True,
                              help='Whether to draw the reconstructed/predicted skeleton or not.')
parser.add_argument('--draw_pred_bbox',type=bool,default=False,
                              help='Whether to draw the bounding box of the reconstructed/predicted trajectories '
                                   'or not.')
parser.add_argument('--person_id', type=int, help='Draw only a specific person in the video.')
parser.add_argument('--draw_local_skeleton', action='store_true',
                              help='If specified, draw local skeletons on a white background. It must be used '
                                   'in conjunction with --person_id, since it is only possible to visualise '
                                   'one pair (ground-truth, reconstructed/predicted) of local skeletons.')
parser.add_argument('--write_dir', default='./visualise', type=str,
                              help='Directory to write rendered frames. If the specified directory does not '
                                   'exist, it will be created.')
parser.add_argument('--generate_gif',type=bool,default=False,
                              help='Render gif from the prediction frames.')
parser.add_argument('--scale',type=int,default=1,
                              help='scale of frames.')


def prepare_keypoints(keypoints):
    keypoints = keypoints * 8
    min_x = min([k[0] for k in keypoints if k[0]!=0])
    min_y = min([k[1] for k in keypoints if k[1]!=0])
    
    max_x = max([k[0] for k in keypoints if k[0]!=0])
    max_y = max([k[1] for k in keypoints if k[1]!=0])
    
    n = 800 / (max_x-min_x)
    
    keypoints = keypoints * n
    
    new_ks = []
    for x,y in keypoints:
        if 0 in (x, y):
            new_ks.append((x,y))
        else: 
            new_ks.append((x-(min_x*n)+40.,y-(min_y*n)+40.))
    keypoints = new_ks
    frame = np.full((math.floor((max_y-min_y)*n+80.),math.floor((max_x-min_x)*n+80.)),fill_value=255, dtype=np.single)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    return keypoints,frame


def draw_skeleton(frame, keypoints, colour, dotted=False, scale=4, scale_vis=False):
    connections = [(5, 6), (5, 11), (6, 12), (11, 12),
                   (0, 1), (0, 2), (1, 3), (2, 4),
                   (5, 7), (7, 9), (6, 8), (8, 10),
                   (11, 13), (13, 15), (12, 14), (14, 16),
                   (0, 5), (0, 6)]
    
    keypoints = keypoints * scale
    
    if scale_vis:
        line_thickness=4*10
        circle_thickness = -1
        radius =3*12
    else:
        line_thickness=4
        circle_thickness = 2
        radius =3 


    for i,(keypoint_id1, keypoint_id2) in enumerate(connections):
        x1, y1 = keypoints[keypoint_id1]
        x2, y2 = keypoints[keypoint_id2]
        if 0 in (x1, y1, x2, y2):
            continue
        pt1 = int(round(x1)), int(round(y1))
        pt2 = int(round(x2)), int(round(y2))
        if dotted:
            draw_line(frame, pt1=pt1, pt2=pt2, color=COLOURS[connections[i]], thickness=line_thickness, gap=5)
        else:
            cv2.line(frame, pt1=pt1, pt2=pt2, color=COLOURS[connections[i]], thickness=line_thickness)
    
    for i, (x, y) in enumerate(keypoints):
        if 0 in (x, y):
            continue
        center = int(round(x)), int(round(y))
        cv2.circle(frame, center=center, radius=radius, color=COLOURS_POINTS[i], thickness=circle_thickness)

    return None

def draw_rect(img, pt1, pt2, color, thickness=1, style='dotted'):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    draw_poly(img, pts, color, thickness, style)

def draw_line(img, pt1, pt2, color, thickness=1, style='dotted', gap=10):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1

def draw_poly(img, pts, color, thickness=1, style='dotted'):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        draw_line(img, s, e, color, thickness, style)

def compute_simple_bounding_box(skeleton):
    x = skeleton[::2]
    x = np.where(x == 0.0, np.nan, x)
    left, right = int(round(np.nanmin(x))), int(round(np.nanmax(x)))
    y = skeleton[1::2]
    y = np.where(y == 0.0, np.nan, y)
    top, bottom = int(round(np.nanmin(y))), int(round(np.nanmax(y)))

    return left, right, top, bottom

def render_trajectories_skeletons(args):
    try:
        os.makedirs(args.write_dir)
    except OSError:
        print(f' \n directory for the images already exists. IMAGES WILL BE REWRITTEN!!! \n')
        pass

    frames_path = args.frames
    gt_trajectories_path = args.gt_trajectories
    draw_gt_skeleton = args.draw_gt_skeleton
    draw_gt_bounding_box = args.draw_gt_bbox
    trajectories_path = args.trajectories
    draw_trajectories_skeleton = args.draw_pred_skeleton
    draw_trajectories_bounding_box = args.draw_pred_bbox
    specific_person_id = args.person_id
    draw_local_skeleton = args.draw_local_skeleton

    if gt_trajectories_path is None and trajectories_path is None:
        raise ValueError('At least one of --ground_truth_trajectories or --trajectories must be specified.')

    if not any([draw_gt_skeleton, draw_gt_bounding_box, draw_trajectories_skeleton, draw_trajectories_bounding_box]):
        raise ValueError('At least one of --draw_ground_truth_trajectories_skeleton, '
                         '--draw_ground_truth_trajectories_bounding_box, --draw_trajectories_skeleton or '
                         '--draw_trajectories_bounding_box must be specified.')

    if draw_local_skeleton and specific_person_id is None:
        raise ValueError('If --draw_local_skeleton is specified, a --person_id must be chosen as well.')
    elif draw_local_skeleton:
        draw_gt_skeleton = draw_trajectories_skeleton = True
        draw_gt_bounding_box = draw_trajectories_bounding_box = False


    _render_trajectories_skeletons(args.write_dir, frames_path, gt_trajectories_path, trajectories_path, specific_person_id, scale=args.scale)

    print('Visualisation successfully rendered to %s' % args.write_dir)

    return None

def fill(frames_path, frame_name,scale,ts):
    if ts is None or not ts:
        ts = [None,None]
        ts[0] = cv2.imread(os.path.join(frames_path, frame_name))
        h,w,c = ts[0].shape
        ts[0] = cv2.resize(ts[0], (w*scale,h*scale), interpolation = cv2.INTER_AREA)
        ts[1] = np.full_like(ts[0], fill_value=255)
    return ts

def fill_multi(frames_path, frame_name,scale,ts,person_ids):
    if ts is None:
        ts = {p : [] for p in person_ids}
        for p in ts.keys():
            ts1 = cv2.imread(os.path.join(frames_path, frame_name))
            h,w,c = ts1.shape
            ts1 = cv2.resize(ts1, (w*scale,h*scale), interpolation = cv2.INTER_AREA)
            ts2 = np.full_like(ts1, fill_value=255)
            ts[p] = (ts1,ts2)
    
    for p in person_ids:
        if p not in ts.keys():
            ts1 = cv2.imread(os.path.join(frames_path, frame_name))
            h,w,c = ts1.shape
            ts1 = cv2.resize(ts1, (w*scale,h*scale), interpolation = cv2.INTER_AREA)
            ts2 = np.full_like(ts1, fill_value=255)
            ts[p] = (ts1,ts2)
    return ts
    

def _render_trajectories_skeletons(write_dir, frames_path, gt_trajectories_path, trajectories_path, specific_person_id=None, scale=4):
    
    vid_id = trajectories_path.split('/')[-1]
    w_dirs = [os.path.join(write_dir,'frames',s,vid_id) for s in ['ind_pred','ind_gt','all_pred','all_gt']]
    for d in w_dirs:
        if not os.path.isdir(d):
            os.makedirs(d)
    
    wo_dirs = [os.path.join(write_dir,'trajectories',s,vid_id) for s in ['ind_pred','ind_gt','all_pred','all_gt']]
    for d in wo_dirs:
        if not os.path.isdir(d):
            os.makedirs(d)
    
    frames_names = sorted(os.listdir(frames_path))  # 000.jpg, 001.jpg, ...
    max_frame_id = len(frames_names)
    rendered_pred_frames_all = {}
    rendered_pred_frames_ind = {}
    
    rendered_gt_frames_all = {}
    rendered_gt_frames_ind = {}
    person_ids = []
    
    if trajectories_path is not None:
        trajectories_files_names = sorted(os.listdir(trajectories_path))  # 001.csv, 002.csv, ...
        for trajectory_file_name in trajectories_files_names:
            person_id = int(trajectory_file_name.split('.')[0])
            if specific_person_id is not None and specific_person_id != person_id:
                continue
            print('Drawing skeleton for person_id:',person_id)
            if person_id not in person_ids:
                person_ids.append(person_id)

            colour = COLOURS[person_id % len(COLOURS)]
            

            trajectory = np.loadtxt(os.path.join(trajectories_path, trajectory_file_name), delimiter=',', ndmin=2)
            trajectory_frames = trajectory[:, 0].astype(np.int32)
            trajectory_coordinates = trajectory[:, 1:]

            for frame_id, skeleton_coordinates in zip(trajectory_frames, trajectory_coordinates):
                if frame_id >= max_frame_id:
                    break
                
                frame_ind = cv2.imread(os.path.join(frames_path, frames_names[frame_id]))
                h,w,c = frame_ind.shape
                frame_ind = cv2.resize(frame_ind, (w*scale,h*scale), interpolation = cv2.INTER_AREA)
                blank_frame_ind = np.full_like(frame_ind, fill_value=255)
                
                coords, blank_frame_ind = prepare_keypoints(skeleton_coordinates.reshape(-1, 2))
                
                el = rendered_pred_frames_all.get(frame_id)
                
                if el is not None:
                    frame = el[0]
                    blank_frame = el[1]
                else:
                    frame = frame_ind.copy()
                    blank_frame = np.full_like(frame_ind, fill_value=255)
                
                draw_skeleton(frame, keypoints=skeleton_coordinates.reshape(-1, 2), colour=colour, dotted=False, scale=scale)
                draw_skeleton(frame_ind, keypoints=skeleton_coordinates.reshape(-1, 2), colour=colour, dotted=False, scale=scale)
                
                #height, width = blank_frame_ind.shape[:2]
                #left, right, top, bottom = compute_simple_bounding_box(skeleton_coordinates)
                #bb_center = np.array([(left + right) / 2, (top + bottom) / 2], dtype=np.float32)
                #target_center = np.array([3 * width / 4, height / 2], dtype=np.float32)
                #displacement_vector = target_center - bb_center
                
                draw_skeleton(blank_frame_ind, keypoints=coords,colour=colour, dotted=False, scale=scale, scale_vis=True)
                draw_skeleton(blank_frame, keypoints=skeleton_coordinates.reshape(-1, 2),colour=colour, dotted=False, scale=scale)
                

                rendered_pred_frames_all[frame_id] = (frame,blank_frame)
                if frame_id not in rendered_pred_frames_ind.keys():
                    rendered_pred_frames_ind[frame_id] = {}
                if person_id not in rendered_pred_frames_ind[frame_id].keys():
                    rendered_pred_frames_ind[frame_id][person_id] = []
                rendered_pred_frames_ind[frame_id][person_id] = (frame_ind,blank_frame_ind)

    if gt_trajectories_path is not None:
        gt_trajectories_files_names = sorted(os.listdir(gt_trajectories_path))
        for gt_trajectory_file_name in gt_trajectories_files_names:
            person_id = int(gt_trajectory_file_name.split('.')[0])
            if specific_person_id is not None and specific_person_id != person_id:
                continue

            
            colour = COLOURS[person_id % len(COLOURS)]

            gt_trajectory = np.loadtxt(os.path.join(gt_trajectories_path, gt_trajectory_file_name),
                                       delimiter=',', ndmin=2)
            gt_trajectory_frames = gt_trajectory[:, 0].astype(np.int32)
            gt_trajectory_coordinates = gt_trajectory[:, 1:]

            for frame_id, skeleton_coordinates in zip(gt_trajectory_frames, gt_trajectory_coordinates):
                
                skeleton_is_null = np.any(skeleton_coordinates)
                if not skeleton_is_null:
                    continue
                
                frame_ind = cv2.imread(os.path.join(frames_path, frames_names[frame_id]))
                h,w,c = frame_ind.shape
                frame_ind = cv2.resize(frame_ind, (w*scale,h*scale), interpolation = cv2.INTER_AREA)
                blank_frame_ind = np.full_like(frame_ind, fill_value=255)
                
                coords, blank_frame_ind = prepare_keypoints(skeleton_coordinates.reshape(-1, 2))
                
                el = rendered_gt_frames_all.get(frame_id)
                
                if el is not None:
                    frame = el[0]
                    blank_frame = el[1]
                else:
                    frame = frame_ind.copy()
                    blank_frame = np.full_like(frame_ind, fill_value=255)
                
                draw_skeleton(frame, keypoints=skeleton_coordinates.reshape(-1, 2), colour=colour, dotted=False, scale=scale)
                draw_skeleton(frame_ind, keypoints=skeleton_coordinates.reshape(-1, 2), colour=colour, dotted=False, scale=scale)
                
                #height, width = blank_frame_ind.shape[:2]
                #left, right, top, bottom = compute_simple_bounding_box(skeleton_coordinates)
                #bb_center = np.array([(left + right) / 2, (top + bottom) / 2], dtype=np.float32)
                #target_center = np.array([3 * width / 4, height / 2], dtype=np.float32)
                #displacement_vector = target_center - bb_center
                
                draw_skeleton(blank_frame_ind, keypoints=coords,colour=colour, dotted=False, scale=scale, scale_vis=True)
                draw_skeleton(blank_frame, keypoints=skeleton_coordinates.reshape(-1, 2),colour=colour, dotted=False, scale=scale)
                

                rendered_gt_frames_all[frame_id] = (frame,blank_frame)
                if frame_id not in rendered_gt_frames_ind.keys():
                    rendered_gt_frames_ind[frame_id] = {}
                if person_id not in rendered_gt_frames_ind[frame_id].keys():
                    rendered_gt_frames_ind[frame_id][person_id] = []
                rendered_gt_frames_ind[frame_id][person_id] = (frame_ind,blank_frame_ind)
                
    
    for frame_id, frame_name in tqdm.tqdm(enumerate(frames_names),total=len(frames_names)):
        
        pred_frame_ind = rendered_pred_frames_ind.get(frame_id)
        pred_frame_all = rendered_pred_frames_all.get(frame_id)
        gt_frame_ind = rendered_gt_frames_ind.get(frame_id)
        gt_frame_all = rendered_gt_frames_all.get(frame_id)
        
        
        pred_frame_all = fill(frames_path, frame_name,scale,pred_frame_all)
        pred_frame_ind = fill_multi(frames_path, frame_name,scale,pred_frame_ind,person_ids)
        
        gt_frame_all = fill(frames_path, frame_name,scale,gt_frame_all)
        gt_frame_ind = fill_multi(frames_path, frame_name,scale,gt_frame_ind,person_ids)
                
        
        #cv2.imwrite(os.path.join(w_dirs[0],frame_name), pred_frame_ind[0])
        #cv2.imwrite(os.path.join(w_dirs[1],frame_name), gt_frame_ind[0])
        cv2.imwrite(os.path.join(w_dirs[2],frame_name), pred_frame_all[0])
        cv2.imwrite(os.path.join(w_dirs[3],frame_name), gt_frame_all[0])
        
        #cv2.imwrite(os.path.join(wo_dirs[0],frame_name), pred_frame_ind[1])
        #cv2.imwrite(os.path.join(wo_dirs[1],frame_name), gt_frame_ind[1])
        cv2.imwrite(os.path.join(wo_dirs[2],frame_name), pred_frame_all[1])
        cv2.imwrite(os.path.join(wo_dirs[3],frame_name), gt_frame_all[1])
        
        for person_id in pred_frame_ind.keys():
            pred_frame_ind_pid = pred_frame_ind.get(person_id)
            if not os.path.isdir(os.path.join(w_dirs[0],str(person_id))):
                os.makedirs(os.path.join(w_dirs[0],str(person_id)))
            if not os.path.isdir(os.path.join(wo_dirs[0],str(person_id))):
                os.makedirs(os.path.join(wo_dirs[0],str(person_id)))
            cv2.imwrite(os.path.join(w_dirs[0],str(person_id),frame_name), pred_frame_ind_pid[0])
            cv2.imwrite(os.path.join(wo_dirs[0],str(person_id),frame_name), pred_frame_ind_pid[1])
        
        
        for person_id in gt_frame_ind.keys():
            gt_frame_ind_pid = gt_frame_ind.get(person_id)
            gt_frame_ind_pid = fill(frames_path, frame_name,scale,gt_frame_ind_pid)
            if not os.path.isdir(os.path.join(w_dirs[1],str(person_id))):
                os.makedirs(os.path.join(w_dirs[1],str(person_id)))
            if not os.path.isdir(os.path.join(wo_dirs[1],str(person_id))):
                os.makedirs(os.path.join(wo_dirs[1],str(person_id)))
            cv2.imwrite(os.path.join(w_dirs[1],str(person_id),frame_name), gt_frame_ind_pid[0])
            cv2.imwrite(os.path.join(wo_dirs[1],str(person_id),frame_name), gt_frame_ind_pid[1])
        




def main():
    args = parser.parse_args()

    if not os.path.exists(args.write_dir):
        os.makedirs(args.write_dir)

    render_trajectories_skeletons(args)


if __name__ == '__main__':
    main()
