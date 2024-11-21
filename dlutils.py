import os
import numpy as np
import torch
import torch.nn as nn
import logging
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
#logging.basicConfig(level=logging.INFO)


class PositionalEncoder(nn.Module):
    r'''
    Sine-cosine positional encoder for input points.
    '''
    def __init__(self, d_input: int, n_freqs: int, log_space:bool=False):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [lambda x : x]

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2.0 **torch.linspace(0.0, self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0**(self.n_freqs-1), self.n_freqs)

        # Alternate sin and cosine
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x*freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x*freq))

    def forward(self, x)-> torch.Tensor:
        r"""
        Apply positioanl encoding to input
        """
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)
    

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.fc3(x)
        return x
    
class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)  # Add an extra dimension for features
        out = self.fc(out[:, -1, :])
        return out

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation 
        self.lin = Linear(in_channels, out_channels, bias=False)
        #self.bias = Parameter(torch.empty(bias_dim))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        #self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Add self-loops to the adjacency matrix.

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(1))
        # Linearly transform node feature matrix.
        x = self.lin(x.transpose(-1,-2)).transpose(-1,-2)
        # Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(1), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # Start propagating messages.
        #print('x = {0}'.format(x))
        out = self.propagate(edge_index, x=x, norm=norm)
        #print('out shape = {0}'.format(out))
        # Apply a final bias vector.
        #out += self.bias
        #print(out)
        return out

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation 
        self.lin = Linear(in_channels, out_channels, bias=False)
        #self.bias = Parameter(torch.empty(bias_dim))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        #self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Add self-loops to the adjacency matrix.

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(1))
        # Linearly transform node feature matrix.
        x = self.lin(x.transpose(-1,-2)).transpose(-1,-2)
        # Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(1), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # Start propagating messages.
        #print('x = {0}'.format(x))
        out = self.propagate(edge_index, x=x, norm=norm)
        #print('out shape = {0}'.format(out))
        # Apply a final bias vector.
        #out += self.bias
        #print(out)
        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
    
class GraphResNetBlock(torch.nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super(GraphResNetBlock, self).__init__()
        self.gc = GCNConv(in_features, out_features)
        self.active = torch.nn.Tanh()

    def forward(self, x, edge_index):
        residual = x
        x = self.gc(x, edge_index)
        x = self.active(x)
        x = x + residual
        return x

class GraphResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks):
        super(GraphResNet, self).__init__()
        self.input_layer = nn.Linear(input_dim, output_dim)
        self.leaky_relu = nn.LeakyReLU()
        self.blocks = nn.ModuleList([
            GraphResNetBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(output_dim, input_dim)

    def forward(self, x, edge_index):
        x = self.leaky_relu(self.input_layer(x))
        for block in self.blocks:
            x = block(x, edge_index)
        x = self.output_layer(x)
        return x

def initialize_model(dim_feature_in:int, dim_feature_out:int, edges:torch.Tensor, device:str):
    fitnet = Fitnet(dim_feature_in=dim_feature_in, dim_feature_out=dim_feature_out,edges=edges, device=device)
    return fitnet 

class Fitnet(nn.Module):
    def __init__(self, dim_feature_in:int, dim_feature_out:int, edges:torch.Tensor, device:str):
        super(Fitnet, self).__init__()
        self.mlp = MLP(input_size=dim_feature_in, hidden_size=2048, output_size=7770).to(device=device)
        self.grunet = GRUNet(input_size=7770, hidden_size=2048, num_layers=8, output_size=7770).to(device=device)
        self.graph_net = GraphResNet(input_dim=3, hidden_dim=2590, output_dim=8, num_blocks=16).to(device=device)
        self.edges = edges

    
    def forward(self, x):
        out = self.mlp(x)
        out = self.grunet(out)
        #logging.debug(f' out shape : {out.shape}')
        out = out.view(-1, 2590, 3)
        out = self.graph_net(out, self.edges)
        return out
def vertex_loss_with_laplacian_smooth(y_gt:torch.tensor, y_pred:torch.tensor,
                                      n_gt:torch.tensor, n_pred:torch.tensor,
                                        L:torch.tensor, ratio=[0.5, 0.4, 0.1]):
    laplacian_matrix = L
    vertex_loss = torch.mean((y_gt-y_pred)**2)
    normal_loss = torch.mean((n_gt-n_pred)**2)
    laplacian_vertex = torch.matmul(L, y_gt-y_pred)
    laplacian_loss = ratio * torch.mean(laplacian_vertex**2)
    return ratio[0] * vertex_loss + ratio[1] * normal_loss + ratio[2] * laplacian_loss
if __name__=='__main__':
    input_data = torch.randn(128,4, 7770).to(device='cuda')
    grunet = GRUNet(input_size=7770, hidden_size=128, num_layers=3, output_size=7770).to(device='cuda')
    for i in range(5000):
        output = grunet(input_data)
    logging.debug(f' output shape : {output.shape}')
    print('Done')