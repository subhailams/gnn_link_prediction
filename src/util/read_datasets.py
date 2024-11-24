import os 
import torch
import numpy as np
from torch_sparse import SparseTensor
from torch.nn.init import xavier_uniform_

import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected, degree

import joblib  # Make ogb loads faster...idk
from ogb.linkproppred import PygLinkPropPredDataset

from util.calc_ppr_scores import get_ppr
import pickle
import dgl
from util.sampler import NeighborSampler

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "dataset")
HEART_DIR = os.path.join(DATA_DIR, "heart")


def read_data_ogb(args, device):
    """
    Read data for OGB datasets
    """
    data_obj = {
        "dataset": args.data_name,
    }

    print("Loading all data...")

    dataset = PygLinkPropPredDataset(name=args.data_name)
    data = dataset[0].to(device)
    split_edge = dataset.get_edge_split()

    if "collab" in args.data_name:
        data, split_edge = filter_by_year(data, split_edge)
        data = data.to(device)

    data_obj['num_nodes'] = data.num_nodes
    edge_index = data.edge_index

    if args.data_name != 'ogbl-citation2':
        data_obj['train_pos'] = split_edge['train']['edge'].to(device)
        data_obj['valid_pos'] = split_edge['valid']['edge'].to(device)
        data_obj['valid_neg'] = split_edge['valid']['edge_neg'].to(device)
        data_obj['test_pos'] = split_edge['test']['edge'].to(device)
        data_obj['test_neg'] = split_edge['test']['edge_neg'].to(device)
    else:
        source_edge, target_edge = split_edge['train']['source_node'], split_edge['train']['target_node']
        data_obj['train_pos'] = torch.cat([source_edge.unsqueeze(1), target_edge.unsqueeze(1)], dim=-1).to(device)

        source, target = split_edge['valid']['source_node'],  split_edge['valid']['target_node']
        data_obj['valid_pos'] = torch.cat([source.unsqueeze(1), target.unsqueeze(1)], dim=-1).to(device)
        data_obj['valid_neg'] = split_edge['valid']['target_node_neg'].to(device) 

        source, target = split_edge['test']['source_node'],  split_edge['test']['target_node']
        data_obj['test_pos'] = torch.cat([source.unsqueeze(1), target.unsqueeze(1)], dim=-1).to(device)
        data_obj['test_neg'] = split_edge['test']['target_node_neg'].to(device)

    # Overwrite Val/Test pos sample for ogbl-ppa under HeaRT
    if args.heart and "ppa" in args.data_name:
        with open(f'{HEART_DIR}/{args.data_name}/valid_samples_index.pt', "rb") as f:
            val_pos_ix = torch.load(f)
        with open(f'{HEART_DIR}/{args.data_name}/test_samples_index.pt', "rb") as f:
            test_pos_ix = torch.load(f)

        data_obj['valid_pos'] = data_obj['valid_pos'][val_pos_ix, :]
        data_obj['test_pos'] = data_obj['test_pos'][test_pos_ix, :]

    # Test train performance without evaluating all test samples
    idx = torch.randperm(data_obj['train_pos'].size(0))[:data_obj['valid_pos'].size(0)]
    data_obj['train_pos_val'] = data_obj['train_pos'][idx]

    if hasattr(data, 'x') and data.x is not None:
        data_obj['x'] = data.x.to(device).to(torch.float)
    else:
        data_obj['x'] =  torch.nn.Parameter(torch.zeros(data_obj['num_nodes'], args.dim).to(device))
        xavier_uniform_(data_obj['x'])

    if hasattr(data, 'edge_weight') and data.edge_weight is not None:
        edge_weight = data.edge_weight.to(torch.float)
        data.edge_weight = data.edge_weight.view(-1).to(torch.float)
    else:
        edge_weight = torch.ones(edge_index.size(1)).to(device).float()
    
    data_obj['adj_t'] = SparseTensor.from_edge_index(edge_index, edge_weight.squeeze(-1), [data.num_nodes, data.num_nodes]).to(device)

    # Needed since directed graph
    if args.data_name == 'ogbl-citation2': 
        data_obj['adj_t'] = data_obj['adj_t'].to_symmetric().coalesce()
        data_obj['adj_mask'] = data_obj['adj_t'].to_symmetric().to_torch_sparse_coo_tensor()
    else:
        data_obj['adj_mask'] = data_obj['adj_t'].to_symmetric().to_torch_sparse_coo_tensor()        
    
    # Don't use edge weight. Only 0/1. Not needed for masking
    data_obj['adj_mask'] = data_obj['adj_mask'].coalesce().bool().int()

    if args.use_val_in_test:
        val_edge_index = split_edge['valid']['edge'].t()
        val_edge_index = to_undirected(val_edge_index).to(device)

        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data['full_edge_index'] = full_edge_index.to(device)

        val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=torch.float, device=device)
        val_edge_weight = torch.cat([edge_weight, val_edge_weight], 0).view(-1)
        data_obj['full_adj_t'] = SparseTensor.from_edge_index(full_edge_index, val_edge_weight, [data.num_nodes, data.num_nodes]).to(device)

        # Don't use edge weight. Only 0/1. Not needed for masking
        data_obj['full_adj_mask'] = data_obj['full_adj_t'].to_torch_sparse_coo_tensor()
        data_obj['full_adj_mask'] = data_obj['full_adj_mask'].coalesce().bool().int()
    else:
        data_obj['full_adj_t'] = data_obj['adj_t']
        data_obj['full_adj_mask'] = data_obj['adj_mask']

    data_obj['degree'] = degree(edge_index[0], num_nodes=data_obj['num_nodes']).to(device)
    if args.use_val_in_test:
        data_obj['degree_test'] = degree(full_edge_index[0], num_nodes=data_obj['num_nodes']).to(device)

    ### Load PPR matrix
    print("Reading PPR...", flush=True)
    data_obj['ppr'] = get_ppr(args.data_name, edge_index, data['num_nodes'],
                              0.15, args.eps, False).to(device)  

    if args.use_val_in_test:
        data_obj['ppr'] = get_ppr(args.data_name, data['full_edge_index'], data['num_nodes'],
                                0.15, args.eps, True).to(device)  
    else:
        data_obj['ppr_test'] = data_obj['ppr']

    # Overwrite standard negatives
    if args.heart:
        with open(f'{HEART_DIR}/{args.data_name}/heart_valid_samples.npy', "rb") as f:
            neg_valid_edge = np.load(f)
            data_obj['valid_neg'] = torch.from_numpy(neg_valid_edge).to(device)
        with open(f'{HEART_DIR}/{args.data_name}/heart_test_samples.npy', "rb") as f:
            neg_test_edge = np.load(f)
            data_obj['test_neg'] = torch.from_numpy(neg_test_edge).to(device)

        # For DDI, val/test takes a long time so only use a subset of val
        if "ddi" in args.data_name:
            num_sample = data_obj['valid_pos'].size(0) // 4
            idx = torch.randperm(data_obj['valid_pos'].size(0))[:num_sample].to(device)
            data_obj['valid_pos'] = data_obj['valid_pos'][idx]
            data_obj['valid_neg'] = data_obj['valid_neg'][idx]
            data_obj['train_pos_val'] = data_obj['train_pos_val'][idx]

    return data_obj




def read_data_planetoid(args, device):
    """
    Read all data for the fixed split. Returns as dict
    """
    data_name = args.data_name

    node_set = set()
    train_pos, valid_pos, test_pos = [], [], []
    train_neg, valid_neg, test_neg = [], [], []

    for split in ['train', 'test', 'valid']:
        path = os.path.join(DATA_DIR, data_name, f"{split}_pos.txt")

        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)
            
            node_set.add(sub)
            node_set.add(obj)
            
            if sub == obj:
                continue

            if split == 'train': 
                train_pos.append((sub, obj))
                
            if split == 'valid': valid_pos.append((sub, obj))  
            if split == 'test': test_pos.append((sub, obj))
    
    num_nodes = len(node_set)
    print('# of nodes in ' + data_name + ' is: ', num_nodes)

    for split in ['test', 'valid']:
        path = os.path.join(DATA_DIR, data_name, f"{split}_neg.txt")

        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)

            if split == 'valid': 
                valid_neg.append((sub, obj))               
            if split == 'test': 
                test_neg.append((sub, obj))

    train_edge = torch.transpose(torch.tensor(train_pos), 1, 0)
    edge_index = torch.cat((train_edge,  train_edge[[1,0]]), dim=1)
    edge_weight = torch.ones(edge_index.size(1))
          
    train_pos_tensor = torch.tensor(train_pos)

    valid_pos = torch.tensor(valid_pos)
    valid_neg =  torch.tensor(valid_neg)

    test_pos =  torch.tensor(test_pos)
    test_neg =  torch.tensor(test_neg)

    idx = torch.randperm(train_pos_tensor.size(0))
    idx = idx[:valid_pos.size(0)]
    train_val = train_pos_tensor[idx]

    feature_embeddings = torch.load(os.path.join(DATA_DIR, data_name, "gnn_feature"))
    feature_embeddings = feature_embeddings['entity_embedding']

    data = {"dataset": args.data_name}
    data['edge_index'] = edge_index.to(device)
    data['num_nodes'] = num_nodes

    data['train_pos'] = train_pos_tensor.to(device)
    data['train_pos_val'] = train_val.to(device)

    data['valid_pos'] = valid_pos.to(device)
    data['valid_neg'] = valid_neg.to(device)
    data['test_pos'] = test_pos.to(device)
    data['test_neg'] = test_neg.to(device)

    data['x'] = feature_embeddings.to(device)

    data['adj_t'] = SparseTensor.from_edge_index(edge_index, edge_weight, [num_nodes, num_nodes]).to(device)
    data['full_adj_t'] = data['adj_t'].to(device)

    data['adj_mask'] = data['adj_t'].to_torch_sparse_coo_tensor()
    data['full_adj_mask'] = data['adj_mask'] = data['adj_mask'].coalesce()

    ### Degree of nodes
    data['degree'] = degree(data['edge_index'][0], num_nodes=data['num_nodes']).to(device)

    ### Load PPR Matrix
    data['ppr'] = get_ppr(args.data_name, data['edge_index'], data['num_nodes'],
                          0.15, args.eps, False).to(device)
    data['ppr_test'] = data['ppr']

    # Overwrite standard negative
    if args.heart:
        with open(f'{HEART_DIR}/{args.data_name}/heart_valid_samples.npy', "rb") as f:
            neg_valid_edge = np.load(f)
            data['valid_neg'] = torch.from_numpy(neg_valid_edge)
        with open(f'{HEART_DIR}/{args.data_name}/heart_test_samples.npy', "rb") as f:
            neg_test_edge = np.load(f)
            data['test_neg'] = torch.from_numpy(neg_test_edge)

    return data



    
def filter_by_year(data, split_edge, year=2007):
    """
    From BUDDY code

    remove edges before year from data and split edge
    @param data: pyg Data, pyg SplitEdge
    @param split_edges:
    @param year: int first year to use
    @return: pyg Data, pyg SplitEdge
    """
    selected_year_index = torch.reshape(
        (split_edge['train']['year'] >= year).nonzero(as_tuple=False), (-1,))
    split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
    split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
    split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]
    train_edge_index = split_edge['train']['edge'].t()
    # create adjacency matrix
    new_edges = to_undirected(train_edge_index, split_edge['train']['weight'], reduce='add')
    new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
    data.edge_index = new_edge_index
    data.edge_weight = new_edge_weight.unsqueeze(-1)
    return data, split_edge

def move_to_device(data_dict, device):
    """
    Moves all tensor values in a dictionary to the specified device.

    Args:
        data_dict (dict): Dictionary containing tensors.
        device (torch.device): Target device (e.g., 'cuda' or 'cpu').

    Returns:
        dict: Dictionary with tensors moved to the target device.
    """
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):  # Check if the value is a tensor
            data_dict[key] = value.to(device)
    return data_dict
    
# def load_pinterest_data(args, device):
#     """
#     Load Pinterest dataset with separate features for user and item nodes, and include `node_type`.
#     """
#     # Load dataset metadata from data.pkl
#     data_info_path = os.path.join(args.dataset_path, "data.pkl")
#     train_g_path = os.path.join(args.dataset_path, "train_g.bin")
    
#     # Load train graph
#     g_list, _ = dgl.load_graphs(train_g_path)
#     train_graph = g_list[0]

#     # Check edge types
#     print("Edge types in the graph:", train_graph.etypes)

#     # Combine edges for all edge types
#     all_edges = []
#     for etype in train_graph.etypes:
#         print(f"Processing edge type: {etype}")
#         heads, tails = train_graph.edges(etype=etype, order="srcdst")
#         all_edges.append(torch.stack([heads, tails], dim=0))

#     # Combine edges into a single tensor
#     edge_index = torch.cat(all_edges, dim=1).to(device)
#     print(f"Edge index shape: {edge_index.shape}")

#     # Get number of nodes
#     num_nodes = train_graph.num_nodes()
#     print(f"Number of nodes: {num_nodes}")

#     # Identify user and item nodes
#     num_users = train_graph.num_nodes(ntype="user")  # Number of user nodes
#     num_items = train_graph.num_nodes(ntype="item")  # Number of item nodes
#     print(f"Number of user nodes: {num_users}, Number of item nodes: {num_items}")

#     # Handle item features (`clip_embedding`)
#     if 'clip_embedding' in train_graph.nodes["item"].data:
#         clip_embedding = train_graph.nodes["item"].data['clip_embedding']
        
#         if isinstance(clip_embedding, torch.Tensor):
#             # Handle as tensor directly
#             clip_embedding = clip_embedding.to(device)
#         elif isinstance(clip_embedding, dict):
#             # Concatenate all features in the dictionary
#             clip_embedding = torch.cat([feat.to(device) for feat in clip_embedding.values()], dim=1)
#         else:
#             raise TypeError(f"Unexpected clip_embedding type: {type(clip_embedding)}")

#         # Assign item features
#         item_features = clip_embedding
#     else:
#         # Random features for all items if no clip embedding exists
#         item_features = torch.randn(num_items, args.hidden_dims, device=device)

#     # Aggregate item features for users via the "interacts" edge type
#     user_features = torch.zeros(num_users, item_features.size(1), device=device)
#     interaction_counts = torch.zeros(num_users, 1, device=device)

#     for user_id, item_id in zip(*train_graph.edges(etype="interacts")):
#         user_features[user_id] += item_features[item_id]
#         interaction_counts[user_id] += 1

#     # Normalize user features by interaction counts
#     user_features = user_features / (interaction_counts + 1e-6)  # Avoid division by zero

#     # Construct the full feature matrix
#     x = torch.zeros(num_nodes, item_features.size(1), device=device)
#     x[:num_users] = user_features  # Assign user features
#     x[num_users:] = item_features  # Assign item features

#     # Create node type tensor: 0 for users, 1 for items
#     node_type = torch.zeros(num_nodes, device=device, dtype=torch.long)
#     node_type[num_users:] = 1  # Items are labeled as 1, users as 0

#     # Negative sampling
#     neg_tails = torch.randint(0, num_nodes, (edge_index.size(1),), device=device)
#     valid_neg = torch.stack([heads.to(device), neg_tails[:heads.size(0)]], dim=1).to(device)
#     test_neg = torch.stack([heads.to(device), neg_tails[heads.size(0):]], dim=1).to(device)

#     # Construct sparse adjacency matrix
#     edge_weight = torch.ones(edge_index.size(1)).to(device)
#     adj_t = SparseTensor.from_edge_index(edge_index, edge_weight, [num_nodes, num_nodes]).to(device)

#     # Degree of nodes
#     degree_tensor = degree(edge_index[0], num_nodes=num_nodes).to(device)

#     # Compute PPR matrix
#     print("Computing PPR matrix...")
#     ppr = get_ppr(args.data_name, edge_index, num_nodes, 0.15, args.eps, False).to(device)
#     print(ppr.shape)
#     print(f"Number of training nodes: {edge_index.size(1)}")
#     print(f"Number of nodes in full graph: {num_nodes}")
#     print(f"Feature matrix shape: {x.shape}")

#     # Load validation and test matrices from `data.pkl`
#     with open(data_info_path, "rb") as f:
#         data_info = pickle.load(f)
#         valid_matrix = data_info["val-matrix"].tocoo()
#         test_matrix = data_info["test-matrix"].tocoo()

#     # Convert validation and test matrices to PyTorch tensors
#     valid_pos = torch.tensor(np.vstack((valid_matrix.row, valid_matrix.col)).T, dtype=torch.long, device=device)
#     test_pos = torch.tensor(np.vstack((test_matrix.row, test_matrix.col)).T, dtype=torch.long, device=device)

#     # Negative sampling
#     neg_tails = torch.randint(0, num_nodes, (edge_index.size(1),), device=device)
#     valid_neg = torch.stack([heads.to(device), neg_tails[:heads.size(0)]], dim=1).to(device)
#     test_neg = torch.stack([heads.to(device), neg_tails[heads.size(0):]], dim=1).to(device)

#     # Construct data dictionary
#     data = {
#         "dataset": "pinterest",
#         "num_nodes": num_nodes,
#         "edge_index": edge_index,
#         "train_pos": edge_index.t(),
#         "train_pos_val": valid_pos,
#         "valid_pos": valid_pos,
#         "valid_neg": valid_neg,
#         "test_pos": test_pos,
#         "test_neg": test_neg,
#         "x": x,  # Unified feature matrix
#         "node_type": node_type,  # Node type tensor
#         "adj_t": adj_t,
#         "full_adj_t": adj_t,
#         "adj_mask": adj_t.to_torch_sparse_coo_tensor().coalesce(),
#         "full_adj_mask": adj_t.to_torch_sparse_coo_tensor().coalesce(),
#         "degree": degree_tensor,
#         "ppr": ppr,
#         "ppr_test": ppr,
#     }

#     return data
def load_pinterest_data(args, device):
    import dgl
    import torch
    import os
    import pickle
    import numpy as np
    from torch_sparse import SparseTensor
    from torch_geometric.utils import degree

    data_info_path = os.path.join(args.dataset_path, "data.pkl")
    train_g_path = os.path.join(args.dataset_path, "train_g.bin")
    
    g_list, _ = dgl.load_graphs(train_g_path)
    train_graph = g_list[0].to('cpu')

    if 'clip_embedding' in train_graph.nodes["image"].data:
        clip_embedding = train_graph.nodes["image"].data['clip_embedding']
        
        if isinstance(clip_embedding, torch.Tensor):
            clip_embedding = clip_embedding.to('cpu')
        elif isinstance(clip_embedding, dict):
            clip_embedding = {k: v.to('cpu') for k, v in clip_embedding.items()}
            clip_embedding = torch.cat(list(clip_embedding.values()), dim=1)
        else:
            raise TypeError(f"Unexpected clip_embedding type: {type(clip_embedding)}")
        
        zero_vector = torch.zeros_like(clip_embedding[0])
        valid_items_mask = ~torch.all(clip_embedding == zero_vector, dim=1)
        valid_item_ids = torch.nonzero(valid_items_mask).squeeze()

        train_graph = dgl.node_subgraph(train_graph, {'user': train_graph.nodes('user'), 'image': valid_item_ids})
        clip_embedding = clip_embedding[valid_items_mask]
    else:
        raise ValueError("No CLIP embeddings found in the graph")

    num_users = train_graph.num_nodes(ntype="user")
    num_items = train_graph.num_nodes(ntype="image")
    num_nodes = num_users + num_items
    print(f"Number of user nodes: {num_users}, Number of valid item nodes: {num_items}")

    all_edges = []
    for etype in train_graph.etypes:
        heads, tails = train_graph.edges(etype=etype)
        all_edges.append(torch.stack([heads, tails], dim=0))

    edge_index = torch.cat(all_edges, dim=1).to(device)
    edge_weight = torch.ones(edge_index.size(1), device=device)
    adj_t = SparseTensor.from_edge_index(edge_index, edge_weight, [num_nodes, num_nodes])
    adj_mask = adj_t.to_torch_sparse_coo_tensor().coalesce()
    degree_tensor = degree(edge_index[0], num_nodes=num_nodes).to(device)

    ppr = get_ppr(args.data_name, edge_index, num_nodes, 0.15, args.eps, False).to(device)

    x = torch.zeros(num_nodes, clip_embedding.size(1), device=device)
    x[num_users:] = clip_embedding

    node_type = torch.zeros(num_nodes, dtype=torch.long, device=device)
    node_type[num_users:] = 1

    with open(data_info_path, "rb") as f:
        data_info = pickle.load(f)
        valid_matrix = data_info["val-matrix"].tocoo()
        test_matrix = data_info["test-matrix"].tocoo()

    valid_pos = torch.tensor(np.vstack((valid_matrix.row, valid_matrix.col)).T, dtype=torch.long, device=device)
    test_pos = torch.tensor(np.vstack((test_matrix.row, test_matrix.col)).T, dtype=torch.long, device=device)

    neg_tails = torch.randint(0, num_nodes, (edge_index.size(1),), device=device)
    valid_neg = torch.stack([edge_index[0][:edge_index.size(1)//2], neg_tails[:edge_index.size(1)//2]], dim=1)
    test_neg = torch.stack([edge_index[0][edge_index.size(1)//2:], neg_tails[edge_index.size(1)//2:]], dim=1)

    data = {
        "dataset": "pinterest",
        "num_nodes": num_nodes,
        "edge_index": edge_index,
        "train_pos": edge_index.t(),
        "train_pos_val": valid_pos,
        "valid_pos": valid_pos,
        "valid_neg": valid_neg,
        "test_pos": test_pos,
        "test_neg": test_neg,
        "x": x,
        "node_type": node_type,
        "adj_t": adj_t,
        "full_adj_t": adj_t,
        "adj_mask": adj_mask,
        "full_adj_mask": adj_mask,
        "degree": degree_tensor,
        "ppr": ppr,
    }

    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device)
        elif isinstance(value, SparseTensor):
            data[key] = value.to(device)

    return data
