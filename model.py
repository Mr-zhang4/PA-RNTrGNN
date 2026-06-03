import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import to_sparse_tensor
from dgl.nn.pytorch import GATConv
#N_ROAD=4388
N_ROAD=2613
#N_ROAD=1879
#N_ROAD=1267

def plot_constraint_matrix(constraint_mat, rn, filename="constraint_matrix.png"):
    """可视化创新约束矩阵"""
    plt.figure(figsize=(12, 10))

    # 创建热力图
    ax = sns.heatmap(
        constraint_mat.cpu().numpy(),
        cmap="viridis",
        square=True,
        cbar_kws={"shrink": 0.8}
    )

    # 设置标题和标签
    plt.title("Innovative Road Constraint Matrix", fontsize=16)
    plt.xlabel("Road Segment ID", fontsize=12)
    plt.ylabel("Road Segment ID", fontsize=12)

    # 保存高质量图片
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved constraint matrix visualization to {filename}")

class MyGAT(nn.Module):
    def __init__(self, node_input_dim, node_hidden_dim, num_layers=2, num_heads=4, last_activate=False):
        super(MyGAT, self).__init__()
        self.hid_dim = node_hidden_dim
        assert node_hidden_dim % num_heads == 0
        self.layers = nn.ModuleList(
                [
                    GATConv(
                        in_feats= node_input_dim if i == 0 else node_hidden_dim,
                        out_feats=node_hidden_dim // num_heads,
                        num_heads=num_heads,
                        feat_drop=0.0,
                        attn_drop=0.0,
                        residual=False,
                        activation=F.leaky_relu if i + 1 < num_layers or last_activate else None,
                    )
                    for i in range(num_layers)
                ]
        )

    def forward(self, g, n_feat):
        for _, layer in enumerate(self.layers):
            #print(f"in gat, shape of n_feat is {n_feat.shape}")
            n_feat = layer(g, n_feat)
            #print(f"done here")
            n_feat = n_feat.reshape(-1, self.hid_dim)
        return n_feat


def normalize_adj(adj, mode='random walk'):
    # mode: 'random walk', 'aggregation'
    if mode == 'random walk': # for T. avg weight for sending node
        deg = np.sum(adj, axis=1).astype(np.float32)
        inv_deg = np.reciprocal(deg, out=np.zeros_like(deg), where=deg!=0)
        D_inv = np.diag(inv_deg)
        normalized_adj = np.matmul(D_inv, adj)
    if mode == 'aggregation': # for W. avg weight for receiving node
        deg = np.sum(adj, axis=0).astype(np.float32)
        inv_deg = np.reciprocal(deg, out=np.zeros_like(deg), where=deg!=0)
        D_inv = np.diag(inv_deg)
        normalized_adj = np.matmul(adj, D_inv)
    return normalized_adj


def graph_propagation_sparse(x, A, grid_output, hop=10, dual=False, mat=False):
    # sparse version
    # x: graph signal vector. tensor. (n_road)
    # A: adjacency matrix. tranposed. sparse_tensor. (n_road, n_road)
    # hop: # propagation steps
    # output: propagation result. tensor. (n_road, hop+1)
    
    x = x.permute(1, 0)
    size = x.size(0)
    if not dual:
        if mat:
            A = A.unsqueeze(0).repeat(size, 1, 1)
        else:
            A = A.permute(2, 0, 1)
    else:
        A = A.unsqueeze(0).repeat(size, 1, 1)
    #print(f"in paropagation {x.shape}, {A.shape}, {grid_output.shape}")
    #print(f"The type of A is {type(A)}")

    y = x.unsqueeze(2)
    X = y
    
    #print(f"the X, y shape {X.shape}, {y.shape}")
    if dual: # dual random walk
        for i in range(hop):
            y_down = A.bmm(X) # downstream
            y_up = A.transpose(1, 2).bmm(X) # upstream
            X = torch.cat([y, y_down, y_up], dim=2)
    else: # downstream random walk only
        #print(f"The shape of y is {y.shape}")
        for i in range(hop):
            y = torch.bmm(A, y)
            X = torch.cat([X, y], dim=2)
    #print(f"The out propagation shape is {X.shape}")
    return X


from torch.nn import Parameter
from torch.nn import init
import math

class ChannelFullyConnected(nn.Module):
        
    __constants__ = ['bias', 'in_features', 'channels']

    def __init__(self, in_features, channels, bias=True):
        super(ChannelFullyConnected, self).__init__()
        self.in_features = in_features
        self.channels = channels
        self.weight = Parameter(torch.Tensor(channels, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return torch.mul(input, self.weight).sum(dim=2) + self.bias

    def extra_repr(self):
        return 'in_features={}, channels={}, bias={}'.format(
            self.in_features, self.channels, self.bias is not None
        )
    

class ChannelAttention(nn.Module):
        
    __constants__ = ['bias', 'in_features', 'channels']

    def __init__(self, in_features, out_features, channels, bias=True):
        super(ChannelAttention, self).__init__()
        self.in_features = in_features
        self.channels = channels
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(channels, in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(channels, out_features)) # out_features=1+demand_hop
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input): # input: (history_window, channels=n_road, in_features=1+status_hop)
        #print(f"The shape of input is {input.shape}")
        return torch.mul(input, self.weight).sum(dim=3) + self.bias

    def extra_repr(self):
        return 'in_features={}, out_features={}, channels={}, bias={}'.format(
            self.in_features, self.out_features, self.channels, self.bias is not None
        )
    
    
class Model_TrGNN(nn.Module):
    # TrGNN.
    
    def __init__(self, rn_grid, grid_num, input_size=1, output_size=1, demand_hop=75, status_hop=3):
        super(Model_TrGNN, self).__init__()
        
        self.rn_grid = rn_grid
        self.grid_num = grid_num # TODO
        self.pad_grid, _  = self.merge(self.rn_grid)
        self.input_size = input_size
        self.output_size = output_size
        self.demand_hop = demand_hop
        self.status_hop = status_hop
        self.id_emb_dim = 64
        self.id_size = N_ROAD
        self.grid_id = nn.Parameter(torch.rand(self.grid_num[0], self.grid_num[1], self.id_emb_dim))
        self.grid_len = torch.tensor([fea.shape[0] for fea in self.rn_grid])
        self.grid = nn.GRU(self.id_emb_dim, self.id_emb_dim)
        
        # attention
        self.attention_layer = ChannelAttention(2**(status_hop+1)-1, demand_hop+1, channels=N_ROAD, bias=True) # channels=n_road
                
        # linear output
        self.output_layer = ChannelFullyConnected(in_features=4+24+1+self.id_emb_dim, channels=N_ROAD) # channels=n_road
    
    def merge(self, sequences):
        lengths = [len(seq) for seq in sequences]
        dim = sequences[0].size(1)
        padded_seqs = torch.zeros(len(sequences), max(lengths), dim)

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        # print(f"The shape of paddd is {padded_seqs.shape}")
        return padded_seqs, lengths

    def forward(self, X, T, W, h_init, W_norm, ToD, DoW):
        # X: graph signal. normalized. tensor: (history_window, n_road)
        # T: trajectory transition. normalized. tuple of history_window sparse_tensors: (n_road, n_road)
        # W: weighted road adjacency matrix. # sparse_tensor: (n_road, n_road)
        # h_init: for GRU. (gru_num_layers, n_road, hidden_size)
        # ToD: road-wise one-hot encoding of hour of day. (n_road, 24)
        # DoW: road-wise indicator. 1 for weekdays, 0 for weekends/PHs. (n_road, 1)

        max_grid_len =self.pad_grid.size(1)
        rn_grid = self.pad_grid.reshape(-1, 2)
        a = rn_grid.numpy()[:, 0]
        b = rn_grid.numpy()[:, 1]
        grid_input = self.grid_id[rn_grid.numpy()[:, 0], rn_grid.numpy()[:, 1], :]
        # print(f"TGhe shape before is {grid_input.shape}")
        grid_input = grid_input.reshape(self.id_size, max_grid_len, -1).transpose(0, 1)
        # print(f"the shape of grid input is {grid_input.shape}")
        # print(f"the shape of X is {X.shape}")

        packed_grid_input = nn.utils.rnn.pack_padded_sequence(grid_input, self.grid_len,
                                                      batch_first=False, enforce_sorted=False)
        _, grid_output = self.grid(packed_grid_input)
        grid_output = grid_output.squeeze(0)
        # print(f"The shape of packed input is {grid_output.shape}")

        
        # graph propagation
        H = torch.cat([self.mygraph_propagation_sparse(x, A.transpose(0, 1), hop=self.demand_hop).unsqueeze(0) for x, A in zip(torch.unbind(X, dim=0), T)], dim=0)

        # attention
        S = torch.cat([graph_propagation_sparse(x, W_norm, hop=self.status_hop, dual=True).unsqueeze(0) for x in torch.unbind(X, dim=0)], dim=0)
        att = self.attention_layer(S.unsqueeze(3)) # specify weights and bias for each road segment
        att = F.softmax(att, dim=2) # attention weights across hops sum up to 1. (history_window, n_road, demand_hop+1)
        H = torch.mul(H, att) # (history_window, n_road, demand_hop+1)
        H = torch.sum(H, dim=2) # (history_window, n_road)
        
        # add ToD, DoW features
        H = torch.cat([H.transpose(0, 1), ToD, DoW, grid_output], dim=1) # (n_road, history_window+24+1)
        
        # linear output. specify weights and bias for each road segment
        Y = self.output_layer(H) # (1, 1, n_road)

        return Y.squeeze(0).squeeze(0) 

    
    
class Model_GNN(nn.Module):
    # TrGNN-. Remove trajectory information. Replace T in TrGNN with W_norm.
    
    def __init__(self, input_size=1, output_size=1, demand_hop=75, status_hop=3):
        super(Model_GNN, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.demand_hop = demand_hop
        self.status_hop = status_hop
        
        # attention
        self.attention_layer = ChannelAttention(2**(status_hop+1)-1, demand_hop+1, channels=N_ROAD, bias=True) # channels=n_road
                
        # linear output
        self.output_layer = ChannelFullyConnected(in_features=4+24+1, channels=N_ROAD) # channels=n_road
        

    def forward(self, X, T, W, h_init, W_norm, ToD, DoW):
        # X: graph signal. normalized. tensor: (history_window, n_road)
        # T: trajectory transition. normalized. tuple of history_window sparse_tensors: (n_road, n_road)
        # W: weighted road adjacency matrix. # sparse_tensor: (n_road, n_road)
        # h_init: for GRU. (gru_num_layers, n_road, hidden_size)
        # ToD: road-wise one-hot encoding of hour of day. (n_road, 24)
        # DoW: road-wise indicator. 1 for weekdays, 0 for weekends/PHs. (n_road, 1)
        
        # graph propagation
        H = torch.cat([graph_propagation_sparse(x, W_norm, hop=self.demand_hop).unsqueeze(0) for x in torch.unbind(X, dim=0)], dim=0)

        # attention
        S = torch.cat([graph_propagation_sparse(x, W_norm, hop=self.status_hop, dual=True).unsqueeze(0) for x in torch.unbind(X, dim=0)], dim=0)
        att = self.attention_layer(S.unsqueeze(3)) # specify weights and bias for each road segment
        att = F.softmax(att, dim=2) # attention weights across hops sum up to 1. (history_window, n_road, demand_hop+1)
        H = torch.mul(H, att) # (history_window, n_road, demand_hop+1)
        H = torch.sum(H, dim=2) # (history_window, n_road)
        
        # add ToD, DoW features
        H = torch.cat([H.transpose(0, 1), ToD, DoW], dim=1) # (n_road, history_window+24+1)
        
        # linear output. specify weights and bias for each road segment
        Y = self.output_layer(H) # (1, 1, n_road)

        return Y.squeeze(0).squeeze(0)

class MYModel_TrGNN(nn.Module):
    # TrGNN.
    
    def __init__(self,g, rn_grid, grid_num, input_size=1, output_size=1, demand_hop=75, status_hop=3):
        super(MYModel_TrGNN, self).__init__()
        
        self.rn_grid = rn_grid
        self.grid_num = grid_num # TODO
        self.pad_grid, _  = self.merge(self.rn_grid)
        self.input_size = input_size
        self.output_size = output_size
        self.demand_hop = demand_hop
        self.status_hop = status_hop
        self.id_emb_dim = 64
        self.id_size = N_ROAD
        self.g = g
        self.gnn = MyGAT(128,128)
        self.grid_id = nn.Parameter(torch.rand(self.grid_num[0], self.grid_num[1], self.id_emb_dim))
        self.grid_len = torch.tensor([fea.shape[0] for fea in self.rn_grid])
        self.grid = nn.GRU(self.id_emb_dim, self.id_emb_dim * 2)
        self.mydropout = nn.Dropout(0.5)
        self.gru = nn.GRU(37, 64, batch_first=True, bidirectional=False)

        for name, param in self.grid.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)

        
        # attention
        self.attention_layer = ChannelAttention(2**(status_hop+1)-1, demand_hop+1+128, channels=N_ROAD, bias=True) # channels=n_road
                
        # linear output
        self.output_layer = ChannelFullyConnected(in_features=64, channels=N_ROAD) # channels=n_road
    
    def merge(self, sequences):
        lengths = [len(seq) for seq in sequences]
        dim = sequences[0].size(1)
        padded_seqs = torch.zeros(len(sequences), max(lengths), dim)

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        # print(f"The shape of paddd is {padded_seqs.shape}")
        return padded_seqs, lengths

    def forward(self, X, T, W, h_init, W_norm, ToD, DoW, constraint_mat):
        # X: graph signal. normalized. tensor: (history_window, n_road)
        # T: trajectory transition. normalized. tuple of history_window sparse_tensors: (n_road, n_road)
        # W: weighted road adjacency matrix. # sparse_tensor: (n_road, n_road)
        # h_init: for GRU. (gru_num_layers, n_road, hidden_size)
        # ToD: road-wise one-hot encoding of hour of day. (n_road, 24)
        # DoW: road-wise indicator. 1 for weekdays, 0 for weekends/PHs. (n_road, 1)

        # print(f"The shape of X, T, w_norm, ToD, DoW {X.shape}, {T.shape}, {W_norm.shape}, {ToD.shape}, {DoW.shape}")
        batch_size = X.size(0)
        max_grid_len =self.pad_grid.size(1)
        rn_grid = self.pad_grid.reshape(-1, 2)
        a = rn_grid.numpy()[:, 0]
        b = rn_grid.numpy()[:, 1]
        grid_input = self.grid_id[rn_grid.numpy()[:, 0], rn_grid.numpy()[:, 1], :]
        # print(f"TGhe shape before is {grid_input.shape}")
        grid_input = grid_input.reshape(self.id_size, max_grid_len, -1).transpose(0, 1)
        # print(f"the shape of grid input is {grid_input.shape}")
        # print(f"the shape of X is {X.shape}")

        packed_grid_input = nn.utils.rnn.pack_padded_sequence(grid_input, self.grid_len,
                                                      batch_first=False, enforce_sorted=False)
        _, grid_output = self.grid(packed_grid_input)
        grid_output = grid_output.squeeze(0)
        #road_emb = grid_output.unsqueeze(0).repeat(batch_size, 1,1)
        # print(f"The shape of packed input is {grid_output.shape}")
        # print(f"the shape of X is {X.shape}")
        road_emb = self.mydropout(self.gnn(self.g, grid_output))
        #road_emb = road_emb.unsqueeze(0).repeat(batch_size, 1,1)
        #print(f"the shape of road_emb is {road_emb.shape}")
        road_emb = road_emb.view(1,1,N_ROAD,128).expand(batch_size, 4,N_ROAD, 128)
        
        X = X.permute(1,2,0)
        T = T.permute(1,2,3,0)

        # graph propagation
        #print(f"The X shape is {X.shape}") # [4,2613,32]
        # TODO
        H = torch.cat([graph_propagation_sparse(x, A.transpose(0, 1),grid_output, hop=self.demand_hop).unsqueeze(0) for x, A in zip(torch.unbind(X, dim=0), T)], dim=0)
        H = H.transpose(0,1)
        history_window = H.size(1)
        #H = H.unsqueeze(0)
        H = torch.cat((H, road_emb),dim=-1)
        #print(f"The shape of H is {H.shape}") # (1, 4, 2613, 128)

        # attention
        S = torch.cat([graph_propagation_sparse(x, W_norm,grid_output, hop=self.status_hop, dual=True).unsqueeze(0) for x in torch.unbind(X, dim=0)], dim=0)
        S = S.transpose(0,1)
        #S = torch.cat((S, road_emb.view(1,1,N_ROAD,128).expand(batch_size, history_window,N_ROAD, 128)),dim=-1)
        #print(f"the s shape is {S.shape}")
        att = self.attention_layer(S.unsqueeze(4)) # specify weights and bias for each road segment
        #print(f"the shape of attn is {att.shape}")
        att = F.softmax(att, dim=3) # attention weights across hops sum up to 1. (history_window, n_road, demand_hop+1)
        #print(f"the shape of H {H.shape}, attn {att.shape}")
        H = torch.mul(H, att) # (history_window, n_road, demand_hop+1)
        H = torch.sum(H, dim=3) # (history_window, n_road)
        #print(f"The final shape of H is {H.shape}") # [32, 4, 2613]
        HH = H.permute(1,2,0)
        HH = torch.cat([graph_propagation_sparse(x, constraint_mat,grid_output, hop=1, dual=True).unsqueeze(0) for x in torch.unbind(HH, dim=0)], dim=0)

        HH = HH.permute(1,2,0,3) # [32, 2613,4, 7]
        HH = HH.reshape(batch_size,N_ROAD, -1)
        #print(f"Teh HH is shape is {HH.shape}")
        
        # print(f"the shape of H, ToD, DoW is {H.shape}, {ToD.shape}, {DoW.shape}")
        # add ToD, DoW features
        H = torch.cat([HH, ToD, DoW], dim=2) # (n_road, history_window+24+1)
        
        # linear output. specify weights and bias for each road segment
        #print(f"The shape of H is {H.shape}")
        #Y = self.output_layer(H) # (1, 1, n_road)
        Y,_  = self.gru(H)
        Y = self.output_layer(Y)

        # print(f"The Y is {Y.shape}")
        return Y

    def mygraph_propagation_sparse(self, x, A, grid_output, hop=10, dual=False):
        # x shape (bs, nroad, 1)
        # A shape (bs, nroad, nroad)
        # grid (nroad, id_emb)
        # self.g graph
        #print(f"The shape of mygraph is {x.shape}, {A.shape}, {grid_output.shape}")
        x = x.permute(1, 0)
        size = x.size(0)
        if not dual:
            A = A.permute(2, 0, 1)
        else:
            A = A.unsqueeze(0).repeat(size, 1, 1)
        x = x.unsqueeze(-1)
        x = torch.cat([x, grid_output], dim=2)
        #print(f"The final shape of x is {x.shape}")
        v =  self.gnn(self.g, x)
        #print(f"The shape of v is {v.shape}")
        return v
class MYModel_TrGNN_dynamic_constraint(nn.Module):
    def __init__(self, g, rn_grid, grid_num, input_size=1, output_size=1, 
                 demand_hop=75, status_hop=3):
        super().__init__()
        self.rn_grid = rn_grid
        self.grid_num = grid_num
        self.pad_grid, _ = self.merge(self.rn_grid)
        self.input_size = input_size
        self.output_size = output_size
        self.demand_hop = demand_hop
        self.status_hop = status_hop
        self.id_emb_dim = 64
        self.id_size = N_ROAD
        self.g = g
        self.gnn = MyGAT(128, 128)
        self.grid_id = nn.Parameter(torch.rand(self.grid_num[0], self.grid_num[1], self.id_emb_dim))
        self.grid_len = torch.tensor([fea.shape[0] for fea in self.rn_grid])
        self.grid = nn.GRU(self.id_emb_dim, self.id_emb_dim * 2)
        self.mydropout = nn.Dropout(0.5)
        self.gru = nn.GRU(53, 64, batch_first=True, bidirectional=False)
        
        # 自适应多跳融合权重
        self.adaptive_weights = nn.Parameter(torch.ones(3))  # 用于1跳、2跳、3跳
        self.softmax = nn.Softmax(dim=0)
        
        # 初始化参数
        for name, param in self.grid.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
        
        # 注意力机制
        self.attention_layer = ChannelAttention(2**(status_hop+1)-1, demand_hop+1+128, channels=N_ROAD, bias=True)
                
        # 输出层
        self.output_layer = ChannelFullyConnected(in_features=64, channels=N_ROAD)
        
        # 新增三个线性层统一特征维度
        self.linear_hop1 = nn.Linear(3, 7)   # 1跳输出3维→7维
        self.linear_hop2 = nn.Linear(7, 7)   # 2跳输出7维→7维
        self.linear_hop3 = nn.Linear(15, 7)  # 3跳输出15维→7维
    def merge(self, sequences):
        lengths = [len(seq) for seq in sequences]
        dim = sequences[0].size(1)
        padded_seqs = torch.zeros(len(sequences), max(lengths), dim)

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    def forward(self, X, T, W, h_init, W_norm, ToD, DoW, constraint_mat):
        batch_size = X.size(0)
        max_grid_len = self.pad_grid.size(1)
        rn_grid = self.pad_grid.reshape(-1, 2)
        grid_input = self.grid_id[rn_grid.numpy()[:, 0], rn_grid.numpy()[:, 1], :]
        grid_input = grid_input.reshape(self.id_size, max_grid_len, -1).transpose(0, 1)

        packed_grid_input = nn.utils.rnn.pack_padded_sequence(
            grid_input, self.grid_len, batch_first=False, enforce_sorted=False
        )
        _, grid_output = self.grid(packed_grid_input)
        grid_output = grid_output.squeeze(0)
        road_emb = self.mydropout(self.gnn(self.g, grid_output))
        road_emb = road_emb.view(1, 1, N_ROAD, 128).expand(batch_size, 4, N_ROAD, 128)
        
        X = X.permute(1, 2, 0)
        T = T.permute(1, 2, 3, 0)

        # 需求传播
        H = torch.cat([
            graph_propagation_sparse(x, A.transpose(0, 1), grid_output, hop=self.demand_hop).unsqueeze(0) 
            for x, A in zip(torch.unbind(X, dim=0), T)
        ], dim=0)
        H = H.transpose(0, 1)
        H = torch.cat((H, road_emb), dim=-1)

        # 注意力机制
        S = torch.cat([
            graph_propagation_sparse(x, W_norm, grid_output, hop=self.status_hop, dual=True).unsqueeze(0) 
            for x in torch.unbind(X, dim=0)
        ], dim=0)
        S = S.transpose(0, 1)
        att = self.attention_layer(S.unsqueeze(4))
        att = F.softmax(att, dim=3)
        H = torch.mul(H, att)
        H = torch.sum(H, dim=3)
        HH = H.permute(1, 2, 0)
        
        # 创新点：自适应多跳约束传播
        # 分别计算1跳、2跳、3跳传播
        hop1 = torch.cat([
            graph_propagation_sparse(x, constraint_mat, grid_output, hop=1, dual=True).unsqueeze(0)
            for x in torch.unbind(HH, dim=0)
        ], dim=0)
        
        hop2 = torch.cat([
            graph_propagation_sparse(x, constraint_mat, grid_output, hop=2, dual=True).unsqueeze(0)
            for x in torch.unbind(HH, dim=0)
        ], dim=0)
        
        hop3 = torch.cat([
            graph_propagation_sparse(x, constraint_mat, grid_output, hop=3, dual=True).unsqueeze(0)
            for x in torch.unbind(HH, dim=0)
        ], dim=0)
        hop1 = self.linear_hop1(hop1)
        hop2 = self.linear_hop2(hop2)
        hop3 = self.linear_hop3(hop3)
        # 自适应融合权重
        weights = self.softmax(self.adaptive_weights)
        HH = weights[0] * hop1 + weights[1] * hop2 + weights[2] * hop3
        
        HH = HH.permute(1, 2, 0, 3)
        HH = HH.reshape(batch_size, N_ROAD, -1)
        
        # 添加时间特征
        H = torch.cat([HH, ToD, DoW], dim=2)
        
        # 输出预测
        Y, _ = self.gru(H)
        Y = self.output_layer(Y)

        return Y

    def mygraph_propagation_sparse(self, x, A, grid_output, hop=10, dual=False):
        x = x.permute(1, 0)
        size = x.size(0)
        if not dual:
            A = A.permute(2, 0, 1)
        else:
            A = A.unsqueeze(0).repeat(size, 1, 1)
        x = x.unsqueeze(-1)
        x = torch.cat([x, grid_output], dim=2)
        v = self.gnn(self.g, x)
        return v
class MYModel_TrGNN_optimized(nn.Module):
    def __init__(self, g, rn_grid, grid_num, input_size=1, output_size=1, 
                 demand_hop=75, status_hop=3, ablation_mode='full'):
        super().__init__()
        self.rn_grid = rn_grid
        self.grid_num = grid_num
        self.pad_grid, _ = self.merge(self.rn_grid)
        self.input_size = input_size
        self.output_size = output_size
        self.demand_hop = demand_hop
        self.status_hop = status_hop
        self.id_emb_dim = 64
        self.id_size = N_ROAD
        self.g = g
        self.gnn = MyGAT(128, 128)
        self.grid_id = nn.Parameter(torch.rand(self.grid_num[0], self.grid_num[1], self.id_emb_dim))
        self.grid_len = torch.tensor([fea.shape[0] for fea in self.rn_grid])
        self.grid = nn.GRU(self.id_emb_dim, self.id_emb_dim * 2)
        self.mydropout = nn.Dropout(0.5)
        self.gru = nn.GRU(29, 64, batch_first=True, bidirectional=False)
        self.ablation_mode = ablation_mode       
        # 自适应多跳融合权重
        # 自适应权重处理
        if ablation_mode == 'no_adaptive':
            # 消融实验：无自适应权重（使用固定权重）
            self.adaptive_weights = nn.Parameter(torch.tensor([0.4, 0.4, 0.2]), requires_grad=False)
        else:
            # 完整模型：可学习权重
            self.adaptive_weights = nn.Parameter(torch.tensor([0.6, 0.3, 0.1]))
#        self.adaptive_weights = nn.Parameter(torch.ones(3))  # 用于1跳、2跳、3跳
        self.softmax = nn.Softmax(dim=0)
        
        # 初始化参数
        for name, param in self.grid.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
        
        # 注意力机制
        self.attention_layer = ChannelAttention(2**(status_hop+1)-1, demand_hop+1+128, channels=N_ROAD, bias=True)
                
        # 输出层
        self.output_layer = ChannelFullyConnected(in_features=64, channels=N_ROAD)
        
        # 新增三个线性层统一特征维度
        self.linear_hop1 = nn.Linear(3, 7)   # 1跳输出3维→7维
        self.linear_hop2 = nn.Linear(7, 7)   # 2跳输出7维→7维
        self.linear_hop3 = nn.Linear(15, 7)  # 3跳输出15维→7维
    def merge(self, sequences):
        lengths = [len(seq) for seq in sequences]
        dim = sequences[0].size(1)
        padded_seqs = torch.zeros(len(sequences), max(lengths), dim)

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    def forward(self, X, T, W, h_init, W_norm, ToD, DoW, constraint_mat):
        batch_size = X.size(0)
        max_grid_len = self.pad_grid.size(1)
        rn_grid = self.pad_grid.reshape(-1, 2)
        grid_input = self.grid_id[rn_grid.numpy()[:, 0], rn_grid.numpy()[:, 1], :]
        grid_input = grid_input.reshape(self.id_size, max_grid_len, -1).transpose(0, 1)

        packed_grid_input = nn.utils.rnn.pack_padded_sequence(
            grid_input, self.grid_len, batch_first=False, enforce_sorted=False
        )
        _, grid_output = self.grid(packed_grid_input)
        grid_output = grid_output.squeeze(0)
        road_emb = self.mydropout(self.gnn(self.g, grid_output))
        road_emb = road_emb.view(1, 1, N_ROAD, 128).expand(batch_size, 4, N_ROAD, 128)
        
        X = X.permute(1, 2, 0)
        T = T.permute(1, 2, 3, 0)

        # 需求传播
        H = torch.cat([
            graph_propagation_sparse(x, A.transpose(0, 1), grid_output, hop=self.demand_hop).unsqueeze(0) 
            for x, A in zip(torch.unbind(X, dim=0), T)
        ], dim=0)
        H = H.transpose(0, 1)
        H = torch.cat((H, road_emb), dim=-1)

        # 注意力机制
        S = torch.cat([
            graph_propagation_sparse(x, W_norm, grid_output, hop=self.status_hop, dual=True).unsqueeze(0) 
            for x in torch.unbind(X, dim=0)
        ], dim=0)
        S = S.transpose(0, 1)
        att = self.attention_layer(S.unsqueeze(4))
        att = F.softmax(att, dim=3)
        H = torch.mul(H, att)
        H = torch.sum(H, dim=3)
        HH = H.permute(1, 2, 0)
        # 消融实验处理：无约束传播
        if self.ablation_mode == 'no_constraint':
            # 完全跳过约束传播模块
            HH = HH.reshape(batch_size, N_ROAD, -1)
        else:
            # 正常进行多跳传播
            hop1 = torch.stack([
                graph_propagation_sparse(x, constraint_mat, grid_output, hop=1, dual=True)
                for x in torch.unbind(HH, dim=0)
            ])
            
            hop2 = torch.stack([
                graph_propagation_sparse(x, constraint_mat, grid_output, hop=2, dual=True)
                for x in torch.unbind(HH, dim=0)
            ])
            
            hop3 = torch.stack([
                graph_propagation_sparse(x, constraint_mat, grid_output, hop=3, dual=True)
                for x in torch.unbind(HH, dim=0)
            ])
            hop1 = self.linear_hop1(hop1)
            hop2 = self.linear_hop2(hop2)
            hop3 = self.linear_hop3(hop3)
             # 应用权重
            weights = self.softmax(self.adaptive_weights)
            HH = weights[0] * hop1 + weights[1] * hop2 + weights[2] * hop3
            HH = HH.permute(1, 2, 0, 3)
            HH = HH.reshape(batch_size, N_ROAD, -1)        
        
        # 添加时间特征
        H = torch.cat([HH, ToD, DoW], dim=2)
        
        # 输出预测
        Y, _ = self.gru(H)
        Y = self.output_layer(Y)

        return Y

    def mygraph_propagation_sparse(self, x, A, grid_output, hop=10, dual=False):
        x = x.permute(1, 0)
        size = x.size(0)
        if not dual:
            A = A.permute(2, 0, 1)
        else:
            A = A.unsqueeze(0).repeat(size, 1, 1)
        x = x.unsqueeze(-1)
        x = torch.cat([x, grid_output], dim=2)
        v = self.gnn(self.g, x)
        return v
class MYModel_TrGNN_no_constraint(nn.Module):
    # TrGNN.
    def __init__(self, g, rn_grid, grid_num, input_size=1, output_size=1, 
                 demand_hop=75, status_hop=3, constraint_mode="full"):
    #def __init__(self,g, rn_grid, grid_num, input_size=1, output_size=1, demand_hop=75, status_hop=3,  alpha=0.7):
        #super(MYModel_TrGNN_no_constraint, self).__init__()
        super().__init__()
        #self.alpha = alpha  
        self.constraint_mode = constraint_mode
        self.rn_grid = rn_grid
        self.grid_num = grid_num # TODO
        self.pad_grid, _  = self.merge(self.rn_grid)
        self.input_size = input_size
        self.output_size = output_size
        self.demand_hop = demand_hop
        self.status_hop = status_hop
        self.id_emb_dim = 64
        self.id_size = N_ROAD
        self.g = g
        self.gnn = MyGAT(128,128)
        self.grid_id = nn.Parameter(torch.rand(self.grid_num[0], self.grid_num[1], self.id_emb_dim))
        self.grid_len = torch.tensor([fea.shape[0] for fea in self.rn_grid])
        self.grid = nn.GRU(self.id_emb_dim, self.id_emb_dim * 2)
        self.mydropout = nn.Dropout(0.5)
        self.gru = nn.GRU(29, 64, batch_first=True, bidirectional=False)
        
        for name, param in self.grid.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
        
        # attention
        self.attention_layer = ChannelAttention(2**(status_hop+1)-1, demand_hop+1+128, channels=N_ROAD, bias=True) # channels=n_road
                
        # linear output
        self.output_layer = ChannelFullyConnected(in_features=64, channels=N_ROAD) # channels=n_road
    
    def merge(self, sequences):
        lengths = [len(seq) for seq in sequences]
        dim = sequences[0].size(1)
        padded_seqs = torch.zeros(len(sequences), max(lengths), dim)

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        # print(f"The shape of paddd is {padded_seqs.shape}")
        return padded_seqs, lengths

    def forward(self, X, T, W, h_init, W_norm, ToD, DoW, constraint_mat):
        # 根据模式调整约束矩阵
        if self.constraint_mode == "dist_only":
            # 只保留距离衰减部分
            constraint_mat = constraint_mat * (constraint_mat > 0).float()
        elif self.constraint_mode == "dir_only":
            # 只保留方向相似性部分
            constraint_mat = constraint_mat * (constraint_mat > 0).float()
        # X: graph signal. normalized. tensor: (history_window, n_road)
        # T: trajectory transition. normalized. tuple of history_window sparse_tensors: (n_road, n_road)
        # W: weighted road adjacency matrix. # sparse_tensor: (n_road, n_road)
        # h_init: for GRU. (gru_num_layers, n_road, hidden_size)
        # ToD: road-wise one-hot encoding of hour of day. (n_road, 24)
        # DoW: road-wise indicator. 1 for weekdays, 0 for weekends/PHs. (n_road, 1)

        # print(f"The shape of X, T, w_norm, ToD, DoW {X.shape}, {T.shape}, {W_norm.shape}, {ToD.shape}, {DoW.shape}")
        batch_size = X.size(0)
        max_grid_len =self.pad_grid.size(1)
        rn_grid = self.pad_grid.reshape(-1, 2)
        a = rn_grid.numpy()[:, 0]
        b = rn_grid.numpy()[:, 1]
        grid_input = self.grid_id[rn_grid.numpy()[:, 0], rn_grid.numpy()[:, 1], :]
        #print(f"TGhe shape before is {grid_input.shape}")
        grid_input = grid_input.reshape(self.id_size, max_grid_len, -1).transpose(0, 1)
        # print(f"the shape of grid input is {grid_input.shape}")
        # print(f"the shape of X is {X.shape}")

        packed_grid_input = nn.utils.rnn.pack_padded_sequence(grid_input, self.grid_len,
                                                      batch_first=False, enforce_sorted=False)
        _, grid_output = self.grid(packed_grid_input)
        grid_output = grid_output.squeeze(0)
        #road_emb = grid_output.unsqueeze(0).repeat(batch_size, 1,1)
        # print(f"The shape of packed input is {grid_output.shape}")
        # print(f"the shape of X is {X.shape}")
        road_emb = self.mydropout(self.gnn(self.g, grid_output))
        #road_emb = road_emb.unsqueeze(0).repeat(batch_size, 1,1)
        #print(f"the shape of road_emb is {road_emb.shape}")
        road_emb = road_emb.view(1,1,N_ROAD,128).expand(batch_size, 4,N_ROAD, 128)
        
        X = X.permute(1,2,0)
        T = T.permute(1,2,3,0)

        # graph propagation
        #print(f"The X shape is {X.shape}") # [4,2613,32]
        # TODO
        H = torch.cat([graph_propagation_sparse(x, A.transpose(0, 1),grid_output, hop=self.demand_hop).unsqueeze(0) for x, A in zip(torch.unbind(X, dim=0), T)], dim=0)
        H = H.transpose(0,1)
        history_window = H.size(1)
        #H = H.unsqueeze(0)
        H = torch.cat((H, road_emb),dim=-1)
        #print(f"The shape of H is {H.shape}") # (1, 4, 2613, 128)

        # attention
        S = torch.cat([graph_propagation_sparse(x, W_norm,grid_output, hop=self.status_hop, dual=True).unsqueeze(0) for x in torch.unbind(X, dim=0)], dim=0)
        S = S.transpose(0,1)
        #S = torch.cat((S, road_emb.view(1,1,N_ROAD,128).expand(batch_size, history_window,N_ROAD, 128)),dim=-1)
        #print(f"the s shape is {S.shape}")
        att = self.attention_layer(S.unsqueeze(4)) # specify weights and bias for each road segment
        #print(f"the shape of attn is {att.shape}")
        att = F.softmax(att, dim=3) # attention weights across hops sum up to 1. (history_window, n_road, demand_hop+1)
        #print(f"the shape of H {H.shape}, attn {att.shape}")
        H = torch.mul(H, att) # (history_window, n_road, demand_hop+1)
        H = torch.sum(H, dim=3) # (history_window, n_road)
        #print(f"The final shape of H is {H.shape}") # [32, 4, 2613]
        #HH = H.permute(1,2,0)
        HH = H.permute(1,2,0)
        # Ê¹ÓÃÔ¼Êø¾ØÕó½øÐÐ¶àÌø´«²¥
        HH = torch.cat([graph_propagation_sparse(x, constraint_mat, grid_output, hop=2, dual=True).unsqueeze(0) 
                        for x in torch.unbind(HH, dim=0)], dim=0)
        HH = HH.permute(1,2,0,3) 
        HH = HH.reshape(batch_size, N_ROAD, -1)
        #HH = HH.permute(1,2,0,3) # [32, 2613,4, 7]
        #HH = HH.reshape(batch_size,N_ROAD, -1)
        #HH = H.permute(0, 2,1)
        #print(f"Teh HH is shape is {HH.shape}")
        
        # print(f"the shape of H, ToD, DoW is {H.shape}, {ToD.shape}, {DoW.shape}")
        # add ToD, DoW features
        H = torch.cat([HH, ToD, DoW], dim=2) # (n_road, history_window+24+1)
        #H = torch.cat([H.transpose(1,2), ToD, DoW], dim=2) # (n_road, history_window+24+1)
        
        # linear output. specify weights and bias for each road segment
        #print(f"The shape of H is {H.shape}")
        #Y = self.output_layer(H) # (1, 1, n_road)
        Y,_  = self.gru(H)
        Y = self.output_layer(Y)

        # print(f"The Y is {Y.shape}")
        return Y

    def mygraph_propagation_sparse(self, x, A, grid_output, hop=10, dual=False):
        # x shape (bs, nroad, 1)
        # A shape (bs, nroad, nroad)
        # grid (nroad, id_emb)
        # self.g graph
        #print(f"The shape of mygraph is {x.shape}, {A.shape}, {grid_output.shape}")
        x = x.permute(1, 0)
        size = x.size(0)
        if not dual:
            A = A.permute(2, 0, 1)
        else:
            A = A.unsqueeze(0).repeat(size, 1, 1)
        x = x.unsqueeze(-1)
        x = torch.cat([x, grid_output], dim=2)
        #print(f"The final shape of x is {x.shape}")
        v =  self.gnn(self.g, x)
        #print(f"The shape of v is {v.shape}")
        return v

class MYModel_TrGNN_no_rn(nn.Module):
    # TrGNN.
    
    def __init__(self,g, rn_grid, grid_num, input_size=1, output_size=1, demand_hop=75, status_hop=3):
        super(MYModel_TrGNN_no_rn, self).__init__()
        
        self.rn_grid = rn_grid
        self.grid_num = grid_num # TODO
        self.pad_grid, _  = self.merge(self.rn_grid)
        self.input_size = input_size
        self.output_size = output_size
        self.demand_hop = demand_hop
        self.status_hop = status_hop
        self.id_emb_dim = 64
        self.id_size = N_ROAD
        self.g = g
        #self.gnn = MyGAT(128,128)
        #self.grid_id = nn.Parameter(torch.rand(self.grid_num[0], self.grid_num[1], self.id_emb_dim))
        #self.grid_len = torch.tensor([fea.shape[0] for fea in self.rn_grid])
        #self.grid = nn.GRU(self.id_emb_dim, self.id_emb_dim * 2)
        self.mydropout = nn.Dropout(0.5)
        self.gru = nn.GRU(53, 64, batch_first=True, bidirectional=False)

        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)

        # attention
        self.attention_layer = ChannelAttention(2**(status_hop+1)-1, demand_hop+1, channels=N_ROAD, bias=True) # channels=n_road
                
        # linear output
        self.output_layer = ChannelFullyConnected(in_features=64, channels=N_ROAD) # channels=n_road
    
    def merge(self, sequences):
        lengths = [len(seq) for seq in sequences]
        dim = sequences[0].size(1)
        padded_seqs = torch.zeros(len(sequences), max(lengths), dim)

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        # print(f"The shape of paddd is {padded_seqs.shape}")
        return padded_seqs, lengths

    def forward(self, X, T, W, h_init, W_norm, ToD, DoW, constraint_mat):
        # X: graph signal. normalized. tensor: (history_window, n_road)
        # T: trajectory transition. normalized. tuple of history_window sparse_tensors: (n_road, n_road)
        # W: weighted road adjacency matrix. # sparse_tensor: (n_road, n_road)
        # h_init: for GRU. (gru_num_layers, n_road, hidden_size)
        # ToD: road-wise one-hot encoding of hour of day. (n_road, 24)
        # DoW: road-wise indicator. 1 for weekdays, 0 for weekends/PHs. (n_road, 1)

        # print(f"The shape of X, T, w_norm, ToD, DoW {X.shape}, {T.shape}, {W_norm.shape}, {ToD.shape}, {DoW.shape}")
        batch_size = X.size(0)
        max_grid_len =self.pad_grid.size(1)
        rn_grid = self.pad_grid.reshape(-1, 2)
        a = rn_grid.numpy()[:, 0]
        b = rn_grid.numpy()[:, 1]
        #grid_input = self.grid_id[rn_grid.numpy()[:, 0], rn_grid.numpy()[:, 1], :]
        # print(f"TGhe shape before is {grid_input.shape}")
        #grid_input = grid_input.reshape(self.id_size, max_grid_len, -1).transpose(0, 1)
        # print(f"the shape of grid input is {grid_input.shape}")
        # print(f"the shape of X is {X.shape}")
        grid_output=None

        #packed_grid_input = nn.utils.rnn.pack_padded_sequence(grid_input, self.grid_len,
                                                      #batch_first=False, enforce_sorted=False)
        #_, grid_output = self.grid(packed_grid_input)
        #grid_output = grid_output.squeeze(0)
        #road_emb = grid_output.unsqueeze(0).repeat(batch_size, 1,1)
        # print(f"The shape of packed input is {grid_output.shape}")
        # print(f"the shape of X is {X.shape}")
        #road_emb = self.mydropout(self.gnn(self.g, grid_output))
        #road_emb = road_emb.unsqueeze(0).repeat(batch_size, 1,1)
        #print(f"the shape of road_emb is {road_emb.shape}")
        #road_emb = road_emb.view(1,1,N_ROAD,128).expand(batch_size, 4,N_ROAD, 128)
        
        X = X.permute(1,2,0)
        T = T.permute(1,2,3,0)

        # graph propagation
        #print(f"The X shape is {X.shape}") # [4,2613,32]
        # TODO
        H = torch.cat([graph_propagation_sparse(x, A.transpose(0, 1),grid_output, hop=self.demand_hop).unsqueeze(0) for x, A in zip(torch.unbind(X, dim=0), T)], dim=0)
        H = H.transpose(0,1)
        history_window = H.size(1)
        #H = H.unsqueeze(0)
        #H = torch.cat((H, road_emb),dim=-1)
        #print(f"The shape of H is {H.shape}") # (1, 4, 2613, 128)

        # attention
        S = torch.cat([graph_propagation_sparse(x, W_norm,grid_output, hop=self.status_hop, dual=True).unsqueeze(0) for x in torch.unbind(X, dim=0)], dim=0)
        S = S.transpose(0,1)
        #S = torch.cat((S, road_emb.view(1,1,N_ROAD,128).expand(batch_size, history_window,N_ROAD, 128)),dim=-1)
        #print(f"the s shape is {S.shape}")
        att = self.attention_layer(S.unsqueeze(4)) # specify weights and bias for each road segment
        #print(f"the shape of attn is {att.shape}")
        att = F.softmax(att, dim=3) # attention weights across hops sum up to 1. (history_window, n_road, demand_hop+1)
        #print(f"the shape of H {H.shape}, attn {att.shape}")
        H = torch.mul(H, att) # (history_window, n_road, demand_hop+1)
        H = torch.sum(H, dim=3) # (history_window, n_road)
        #print(f"The final shape of H is {H.shape}") # [32, 4, 2613]
        HH = H.permute(1,2,0)
        HH = torch.cat([graph_propagation_sparse(x, constraint_mat,grid_output, hop=2, dual=True).unsqueeze(0) for x in torch.unbind(HH, dim=0)], dim=0)
        HH = HH.permute(1,2,0,3) # [32, 2613,4, 7]

        #print(f"The shape of HH is {HH.shape}, H is {H.shape}")
        HH = HH.reshape(batch_size,N_ROAD, -1)

        #print(f"Teh HH is shape is {HH.shape}")
        H = torch.cat([HH, ToD, DoW], dim=2) # (n_road, history_window+24+1)
        
        # print(f"the shape of H, ToD, DoW is {H.shape}, {ToD.shape}, {DoW.shape}")
        # add ToD, DoW features
        #H = torch.cat([HH, ToD, DoW], dim=2) # (n_road, history_window+24+1)
        
        # linear output. specify weights and bias for each road segment
        #print(f"The shape of H is {H.shape}")
        # Y = self.output_layer(H) # (1, 1, n_road)
        Y,_  = self.gru(H)
        Y = self.output_layer(Y)

        # print(f"The Y is {Y.shape}")
        return Y

    def mygraph_propagation_sparse(self, x, A, grid_output, hop=10, dual=False):
        # x shape (bs, nroad, 1)
        # A shape (bs, nroad, nroad)
        # grid (nroad, id_emb)
        # self.g graph
        #print(f"The shape of mygraph is {x.shape}, {A.shape}, {grid_output.shape}")
        x = x.permute(1, 0)
        size = x.size(0)
        if not dual:
            A = A.permute(2, 0, 1)
        else:
            A = A.unsqueeze(0).repeat(size, 1, 1)
        x = x.unsqueeze(-1)
        x = torch.cat([x, grid_output], dim=2)
        #print(f"The final shape of x is {x.shape}")
        v =  self.gnn(self.g, x)
        #print(f"The shape of v is {v.shape}")
        return v
