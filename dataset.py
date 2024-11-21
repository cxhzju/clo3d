import os
import logging
import torch
import numpy as np
from torch.utils.data import Dataset

from utils import read_beta, read_pose, load_obj, Mesh
logging.basicConfig(level=logging.DEBUG)

class FittingSet(Dataset):
    def __init__(self, data_dir:str='/home/cxh/Downloads/dataset/PoseShapeSet') -> None:
        self.data_dir = data_dir
        self.num_instances = 109 # number of instance (betas)
        self.num_pose_seq = 15
        self.pose_table = ['catching_and_throwing_poses', 
                           'jumping_poses', 
                           'kicking_poses',
                           'knocking_poses',
                           'lifting_heavy_poses',
                           'lifting_light_poses',
                           'motorcycle_poses',
                           'normal_jog_poses',
                           'normal_walk_poses',
                           'scamper_poses',
                           'sitting2_poses',
                           'sitting_poses',
                           'throwing_hard_poses',
                           'treadmill_jog_poses',
                           'treadmill_norm_poses']
        template_obj_path = os.path.join(self.data_dir, 'template.obj')
        v, f = load_obj(template_obj_path)
        mesh = Mesh(v, f)
        self.edges = mesh.edges
        self.template_faces = f
        self.template_vertices = v
        
    
    def __len__(self):
        return self.num_instances * self.num_pose_seq
    # Given sequence index, return corresponding shape, pose, obj sequence
    # obtained from dataset
    def __getitem__(self, index):
        instance_idx = index//self.num_pose_seq# index of instances
        pose_idx = index % self.num_pose_seq# index of pose pose_table[i] gives the pose name
        pose_name = self.pose_table[pose_idx]
        pose_seq_dir = os.path.join(self.data_dir, 'instance{0:0>3}'.format(instance_idx), pose_name)

        # Get SMPL Pose and shape parameter
        smpl_pose_path = os.path.join(self.data_dir, pose_name+'.npz')
        smpl_shape_path = os.path.join(self.data_dir, 'betas.npz')

        # Load beta and pose parameters
        smpl_pose_params = read_pose(smpl_pose_path)
        smpl_beta_params = read_beta(smpl_shape_path)[instance_idx]
        pose_length = len(smpl_pose_params)

        vertex_seq_path = os.path.join(pose_seq_dir, 'seqs.npy')
        vertex_seq = np.load(vertex_seq_path)
        return smpl_beta_params, smpl_pose_params, vertex_seq
    
    
if __name__=='__main__':
    fittingset = FittingSet(data_dir='/home/cxh/Downloads/dataset/PoseShapeSet')
    logging.debug(f' length : {len(fittingset)}')
    beta, poses, seq = fittingset[1634]
    logging.debug(f' beta size : {beta.shape}')
    logging.debug(f' poses size : {poses.shape}')
    logging.debug(f' seq size : {seq.shape}')
    print('done')