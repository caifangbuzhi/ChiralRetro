from typing import Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import index_select_ND
                   
class BatchGRU(nn.Module):
    def __init__(self, hidden_size=300):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru  = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, 
                           bidirectional=True)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size), 
                                1.0 / math.sqrt(self.hidden_size))


    def forward(self, node, a_scope):
        hidden = node
        message = F.relu(node + self.bias)
        MAX_atom_len = max([a_size for a_start, a_size in a_scope])
        message_lst = []
        hidden_lst = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_message = message.narrow(0, a_start, a_size)
            cur_hidden = hidden.narrow(0, a_start, a_size)
            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))
            
            cur_message = torch.nn.ZeroPad2d((0,0,0,MAX_atom_len-cur_message.shape[0]))(cur_message)
            message_lst.append(cur_message.unsqueeze(0))
            
        message_lst = torch.cat(message_lst, 0)
        hidden_lst  = torch.cat(hidden_lst, 1)
        hidden_lst = hidden_lst.repeat(2,1,1)
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)
        
        cur_message_unpadding = []
        for i, (a_start, a_size) in enumerate(a_scope):
            cur_message_unpadding.append(cur_message[i, :a_size].view(-1, 2*self.hidden_size))
        cur_message_unpadding = torch.cat(cur_message_unpadding, 0)
        
        message = torch.cat([torch.cat([message.narrow(0, 0, 1), message.narrow(0, 0, 1)], 1), 
                             cur_message_unpadding], 0)
        return message    
    
class CMPNNEncoder(nn.Module):
    def __init__(self, config):
        super(CMPNNEncoder, self).__init__()
        self.atom_fdim = config['n_atom_feat']
        self.bond_fdim = config['n_bond_feat']
        self.hidden_size = config['mpn_size']
        self.bias = False
        self.depth = config['depth']
        self.dropout = config['dropout_mpn']
        self.layers_per_message = 1
        self.undirected = False
        self.atom_messages = config['atom_message']
        self.features_only = False
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.act_func = nn.ReLU()

        input_dim = self.atom_fdim
        self.W_i_atom = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        input_dim = self.bond_fdim
        self.W_i_bond = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
          
        w_h_input_size_atom = self.hidden_size + self.bond_fdim
        self.W_h_atom = nn.Linear(w_h_input_size_atom, self.hidden_size, bias=self.bias)
        
        w_h_input_size_bond = self.hidden_size
        
        
        for depth in range(self.depth-1):
            self._modules[f'W_h_{depth}'] = nn.Linear(w_h_input_size_bond, self.hidden_size, bias=self.bias)
        
        self.W_o = nn.Linear(
                (self.hidden_size)*2,
                self.hidden_size)
        
        self.gru = BatchGRU(self.hidden_size)
        
        self.lr = nn.Linear(self.hidden_size*3, self.hidden_size, bias=self.bias)

        
    def forward(self, graph_tensors: Tuple[torch.Tensor], scopes: Tuple[List]) -> torch.FloatTensor:

        f_atoms, f_bonds, a2b, b2a, b2revb, undirected_b2a = graph_tensors
        a_scope, b_scope = scopes

        input_atom = self.W_i_atom(f_atoms)  
        input_atom = self.act_func(input_atom)
        message_atom = input_atom.clone()
        
        input_bond = self.W_i_bond(f_bonds) 
        message_bond = self.act_func(input_bond)
        input_bond = self.act_func(input_bond)

        for depth in range(self.depth - 1):
            agg_message = index_select_ND(message_bond, a2b)   
            agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
            message_atom = message_atom + agg_message       
            
            rev_message = message_bond[b2revb]  
            message_bond = message_atom[b2a] - rev_message   
            
            message_bond = self._modules[f'W_h_{depth}'](message_bond)
            message_bond = self.dropout_layer(self.act_func(input_bond + message_bond)) 
        
        agg_message = index_select_ND(message_bond, a2b)
        agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]    
        agg_message1 = self.lr(torch.cat([agg_message, message_atom, input_atom], 1))    
        agg_message2 = self.gru(agg_message1, a_scope)  
        atom_hiddens = self.act_func(self.W_o(agg_message2))   
        atom_hiddens1 = self.dropout_layer(atom_hiddens)  
    
        mask = torch.ones(atom_hiddens1.size(0), 1, device=f_atoms.device)
        mask[0, 0] = 0   

        return atom_hiddens1 * mask

class MPNEncoder(nn.Module):    # DMPNN
    """Class: 'MPNEncoder' is a message passing neural network for encoding molecules."""

    def __init__(self, atom_fdim: int, bond_fdim: int, hidden_size: int,
                 depth: int, dropout: float = 0.15, atom_message: bool = False, device = 'cpu'):
        """
        Parameters
        ----------
        atom_fdim: Atom feature vector dimension.
        bond_fdim: Bond feature vector dimension.
        hidden_size: Hidden layers dimension
        depth: Number of message passing steps
        droupout: the droupout rate
        atom_message: 'D-MPNN' or 'MPNN', centers messages on bonds or atoms.
       """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim  
        self.bond_fdim = bond_fdim  
        self.hidden_size = hidden_size   
        self.depth = depth   
        self.dropout = dropout
        self.atom_message = atom_message
        self.device = device

        input_dim = self.atom_fdim if self.atom_message else self.bond_fdim  
        self.w_i = nn.Linear(input_dim, self.hidden_size, bias=False)    
        
        self.tetra_perms = torch.tensor([[0, 1, 2, 3],
                                         [0, 2, 3, 1],
                                         [0, 3, 1, 2],
                                         [1, 0, 3, 2],
                                         [1, 2, 0, 3],
                                         [1, 3, 2, 0],
                                         [2, 0, 1, 3],
                                         [2, 1, 3, 0],
                                         [2, 3, 0, 1],
                                         [3, 0, 2, 1],
                                         [3, 1, 0, 2],
                                         [3, 2, 1, 0]])

        # Update message
        if self.atom_message:   
            self.w_h = nn.Linear(
                self.bond_fdim + self.hidden_size, self.hidden_size)

        self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.W_bs = nn.Linear(hidden_size*4, hidden_size)

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout) 
        # Output
        self.W_o = nn.Sequential(
            nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size), nn.ReLU())

    def get_neighbors(self, l1, l2):
        neighbors = [[] for _ in range(len(l1))]
 
        for atom_idx, edge_indices in enumerate(l1):
            for edge_idx in edge_indices:
                neighbor_atom = l2[edge_idx]
                neighbors[atom_idx].append(neighbor_atom.item())
        
        return torch.FloatTensor(neighbors)

    def filter_rows_and_adjust_tensor(self, input_tensor_2d, input_tensor_1d):
        if input_tensor_2d.dim() != 2:
            raise ValueError("The input two-dimensional tensor must be two-dimensional!")

        if input_tensor_1d.dim() != 1:
            raise ValueError("The input one-dimensional tensor must be one-dimensional!")

        if input_tensor_2d.size(0) != input_tensor_1d.size(0):
            raise ValueError("The number of rows in a two-dimensional tensor must be equal to the length of a one-dimensional tensor!")

        filtered_rows = []
        filtered_1d = []

        for i, row in enumerate(input_tensor_2d):
            non_zero_count = (row != 0).sum().item()
            if non_zero_count == 4:
                filtered_rows.append(row[row != 0])
                filtered_1d.append(input_tensor_1d[i])

        filtered_2d = torch.stack(filtered_rows).to(self.device) if filtered_rows else torch.empty((0,))
        filtered_1d = torch.tensor(filtered_1d, device=self.device) if filtered_1d else torch.empty((0,))

        return filtered_2d, filtered_1d

    def forward(self, graph_tensors: Tuple[torch.Tensor], parity_atoms, mask: torch.Tensor) -> torch.FloatTensor: # DMPNN 
        """
        Forward pass of the graph encoder. Encodes a batch of molecular graphs.

        Parameters
        ----------
        graph_tensors: Tuple[torch.Tensor],
            Tuple of graph tensors - Contains atom features, message vector details, the incoming bond indices of atoms
            the index of the atom the bond is coming from, the index of the reverse bond and the undirected bond index 
            to the beginindex and endindex of the atoms.
        mask: torch.Tensor,
            Masks on nodes
        """
        f_atoms, f_bonds, a2b, b2a, b2revb, undirected_b2a, directed_b2a = graph_tensors
        
        if self.atom_message:
            a2a = b2a[a2b]   
            f_bonds = f_bonds[:, -self.bond_fdim:]
            input = self.w_i(f_atoms)   
        else:
            input = self.w_i(f_bonds)  
 
        message = input
        message_mask = torch.ones(message.size(0), 1, device=message.device)     
        message_mask[0, 0] = 0  # first message is padding

        for depth in range(self.depth - 1): # 
            if self.atom_message:
                nei_a_message = index_select_ND(message, a2a)
                nei_f_bonds = index_select_ND(f_bonds, a2b)
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)
                message = nei_message.sum(dim=1)
                message = self.w_h(message)  
            else:   
                if depth <= (self.depth-1)/2:
                    nei_a_message = index_select_ND(message, a2b)   
                    a_message = nei_a_message.sum(dim=1)  
                    tetra_ids = parity_atoms.nonzero(as_tuple=False).squeeze(1) 
                    neighbors_indices = self.get_neighbors(a2b, b2a).to(self.device)    
                    neighbors_indices = neighbors_indices[tetra_ids]
                    
                    tetra_nei_ids, tetra_ids = self.filter_rows_and_adjust_tensor(neighbors_indices, tetra_ids)
                    if tetra_ids.nelement() != 0:
                        ccw_mask = parity_atoms[tetra_ids] == -1    
                        tetra_nei_ids[ccw_mask] = tetra_nei_ids.clone()[ccw_mask][:, [1, 0, 2, 3]]  
                        
                        edge_ids = torch.cat([tetra_nei_ids.view(1, -1), tetra_ids.repeat_interleave(4).unsqueeze(0)], dim=0)    
                        attr_ids = [torch.where((a == directed_b2a).all(dim=1))[0] for a in edge_ids.t()]
                
                        edge_reps = message[attr_ids, :].view(tetra_nei_ids.size(0), 4, -1)   
                        nei_messages = nn.Dropout(self.dropout)(F.relu(self.W_bs(edge_reps[:, self.tetra_perms, :].flatten(start_dim=2))))
                        nei_messages = nei_messages.sum(dim=-2) / 3.
                        a_message[tetra_ids] = nei_messages
 
                    rev_message = message[b2revb]   
                    message = a_message[b2a] - rev_message    
                
                else:
                    nei_a_message = index_select_ND(message, a2b)    
                    a_message = nei_a_message.sum(dim=1)
                    rev_message = message[b2revb]   
                    message = a_message[b2a] - rev_message
                
            message = self.gru(input, message)  
            message = message * message_mask
            message = self.dropout_layer(message)   

        if self.atom_message:
            nei_a_message = index_select_ND(message, a2a)
        else:
            nei_a_message = index_select_ND(message, a2b)
        a_message = nei_a_message.sum(dim=1)   
        a_input = torch.cat([f_atoms, a_message], dim=1)     
        atom_hiddens = self.W_o(a_input)   

        if mask is None:
            mask = torch.ones(atom_hiddens.size(0), 1, device=f_atoms.device)
            mask[0, 0] = 0   

        return atom_hiddens * mask


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, mask.size(-1), 1)
            mask = mask.unsqueeze(1).repeat(1, scores.size(1), 1, 1)
            scores[~mask.bool()] = float(-9e15)
        scores = torch.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        return scores, output

    def forward(self, x, mask=None):
        bs = x.size(0)
        k = self.k_linear(x).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(x).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(x).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        scores, output = self.attention(q, k, v, mask)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = output + x
        output = self.layer_norm(output)
        return scores, output.squeeze(-1)


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.ReLU(),
            nn.Linear(d_model*2, d_model),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        output = self.net(x)
        return self.layer_norm(x + output)


class Global_Attention(nn.Module):
    def __init__(self, d_model, heads, n_layers=1, dropout=0.1):
        super(Global_Attention, self).__init__()
        self.n_layers = n_layers
        att_stack = []
        pff_stack = []
        for _ in range(n_layers):
            att_stack.append(MultiHeadAttention(heads, d_model, dropout))
            pff_stack.append(FeedForward(d_model, dropout))
        self.att_stack = nn.ModuleList(att_stack)
        self.pff_stack = nn.ModuleList(pff_stack)

    def forward(self, x, mask):
        scores = []
        for n in range(self.n_layers):
            score, x = self.att_stack[n](x, mask)
            x = self.pff_stack[n](x)
            scores.append(score)
        return scores, x

