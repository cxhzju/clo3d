import os
from time import sleep
from typing import Optional, Tuple, List
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from random import shuffle
import logging
import tqdm

from dataset import FittingSet
from argparser import parse_arguments, read_config_file
from dlutils import PositionalEncoder, initialize_model
from dlutils import vertex_loss_with_laplacian_smooth
from utils import Mesh

def chunkify(betas:np.ndarray, thetas:np.ndarray, seqs:np.ndarray, chunk_size)->Tuple[np.ndarray,np.ndarray,np.ndarray]:
    seq_length = len(thetas)
    coord = np.lib.stride_tricks.sliding_window_view(np.asarray(range(seq_length),dtype=np.int32), (chunk_size,))
    #logging.debug(f' coord: {coord}')
    chunked_betas = np.asarray([betas[e] for e in coord])
    chunked_thetas = np.asarray([thetas[e] for e in coord])
    chunked_seqs = np.asarray([seqs[e] for e in coord])
    return chunked_betas, chunked_thetas, chunked_seqs
#def initialize_model()

def batchify(data:np.ndarray, batch_size:int)->List:
    num_data = data.shape[0]
    num_batches = num_data // batch_size
    batches = np.array_split(data, num_batches)
    return batches


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()
    config_data = read_config_file(args)

    dataset_dir = config_data['dataset']['dataset_dir']
    device = config_data['device']

    n_epoch = config_data['training']['n_epoch']
    chunk_size = config_data['training']['chunk_size']
    batch_size = config_data['training']['batch_size']
    lr = config_data['training']['lr']

    dim_beta = config_data['dim_beta'] # Dimension of SMPL shape parameter 10
    dim_theta = config_data['dim_theta'] # Dimensionof SMPL pose parameter 72
    n_pos_freqs = config_data['n_pos_freqs']
    n_shape_freqs = config_data['n_shape_freqs']
    #logging.debug(f' dataset dir : {dataset_dir}')

    # Load dataset
    fitting_set = FittingSet(data_dir=dataset_dir)
    edges = torch.tensor(fitting_set.edges, dtype=torch.int64, device=device)
    template_vertices = torch.tensor(fitting_set.template_vertices, dtype=torch.float32, device=device)
    template_faces = fitting_set.template_faces
    # get random indices from fitting set
    dataset_length = len(fitting_set)
    #dataset_length = 50
    data_indices = [i for i in range(dataset_length)] 
    
    # Initialize positional encoder
    shape_encoder = PositionalEncoder(d_input=dim_beta, n_freqs=n_shape_freqs)
    pose_encoder = PositionalEncoder(d_input=dim_theta, n_freqs=n_pos_freqs)
    # Initialize training model
    fitnet = initialize_model(dim_feature_in=1722, dim_feature_out=7770, edges=edges, device=device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(fitnet.parameters(), lr=lr)
    for epoch in range(n_epoch): 
        shuffle(data_indices) # shuffle data at each epoch
        tqdm_data_indices = tqdm.tqdm(data_indices)
        print('Processing epoch : {0}'.format(epoch))
        for idx in tqdm_data_indices:
            beta, thetas, seqs = fitting_set[idx]
            betas = np.expand_dims(beta, axis=0)
            betas = np.repeat(betas, repeats=len(thetas), axis=0)

            seq_length, num_vertices, num_channels = seqs.shape
            #logging.debug(f' thetas shape : {thetas.shape}')
            #logging.debug(f' betas shape : {betas.shape}')
            chunked_betas, chunked_thetas, chunked_seqs = chunkify(betas, thetas, seqs, chunk_size)
            #logging.debug(f' chunked betas shape : {chunked_betas.shape}')
            #logging.debug(f' chunked thetas shape : {chunked_thetas.shape}')
            #logging.debug(f' chunked seqs shape : {chunked_seqs.shape}')
            # mini-batch data
            batched_betas = batchify(chunked_betas, batch_size=batch_size)
            batched_thetas= batchify(chunked_thetas, batch_size=batch_size)
            batched_seqs = batchify(chunked_seqs, batch_size=batch_size)
            
            for batch_idx in range(len(batched_betas)):
                optimizer.zero_grad()   
                beta = batched_betas[batch_idx]
                theta = batched_thetas[batch_idx]
                seq = torch.tensor(batched_seqs[batch_idx][:,0,:,:], dtype=torch.float32, device=device)
                beta = torch.tensor(beta,dtype=torch.float32).to(device=device)
                theta= torch.tensor(theta, dtype=torch.float32).to(device=device)
                encoded_betas = shape_encoder(beta)
                encoded_thetas = pose_encoder(theta)
                #logging.debug(f' encoded betas : {encoded_betas.shape}')
                #logging.debug(f' encoded thetas: {encoded_thetas.shape}')
                encoded_smpl_params = torch.cat([encoded_betas, encoded_thetas], dim=-1)
                #logging.debug(f' batched seqs: {batched_seqs.shape}')
                y = fitnet(encoded_smpl_params) + template_vertices
               
                loss = criterion(y,seq)
                tqdm_data_indices.set_description(f' loss: {loss}')
                loss.backward()
                optimizer.step()
                #y = y.view(-1, 2590, 3)
                