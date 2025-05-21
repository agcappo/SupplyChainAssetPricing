

# In layers, use TransformerConv
class GNNModel(nn.Module):
    def __init__(self, num_features, embedding_dim, num_heads=2, num_layers=2, pca_dim=5):
        super(GNNModel, self).__init__()
        self.layers = nn.ModuleList()
        self.bn = nn.BatchNorm1d(num_features)
        mid_dim = 40
        self.pca_dim = pca_dim
        self.layer_type == 'Transformer'
        self.bn_concat = nn.BatchNorm1d(pca_dim+1)  # concat_dim = scalar_pred_dim + pca_factors_dim
        for i in range(num_layers):
            in_channels = num_features if i == 0 else mid_dim
            out_channels = embedding_dim if i == num_layers - 1 else mid_dim
            self.layers.append(TransformerConv(in_channels, out_channels, heads=num_heads, concat=False, dropout=0.5))
            
        self.supply_chain_factor = nn.Linear(embedding_dim, 1)                 # scalar output from embedding
        self.out = nn.Linear(1 + pca_dim, 1)                    # final scalar using PCA + previous scalar

    def forward(self, x, edge_index, pca_factors):
        x = self.bn(x)
        for layer in self.layers:
            x = F.elu(layer(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)

        x_supply_chain_factor = self.supply_chain_factor(x)                             # shape: (num_nodes, 1)
        x_concat = torch.cat([F.elu(x_supply_chain_factor), pca_factors], dim=1)      
        x_concat = self.bn_concat(x_concat) # normalize the concatenated output jointly
        yhat = self.out(x_concat)                        # shape: (num_nodes, 1)

        return yhat, x_concat

from torch_geometric.data import Data

def prepare_graph_data(data, feature_cols, response_col, time_col='yyyymm', edge_weight_col=None):
    """
    Prepare graph data per month, supporting directed and weighted edges.
    """
    # Create a global node map (consistent node indexing across all months)
    unique_keys = pd.concat([data['gvkey'], data['supplier_or_customer_gvkey']]).dropna().drop_duplicates()
    global_node_map = pd.Series(data=np.arange(len(unique_keys)), index=unique_keys).to_dict()
    num_nodes = len(global_node_map)

    graphs_per_month = {}

    for time, group in data.groupby(time_col):
        x = np.random.randn(num_nodes, len(feature_cols)) / 100
        y = torch.zeros(num_nodes)

        edges_df = group[['gvkey', 'supplier_or_customer_gvkey', 'supplier_or_customer_flag']].copy()
        edges_df['gvkey_idx'] = edges_df['gvkey'].map(global_node_map)
        edges_df['partner_idx'] = edges_df['supplier_or_customer_gvkey'].map(global_node_map)
        edges_df = edges_df.dropna(subset=['gvkey_idx', 'partner_idx'])

        # Directed edges: 0 = supplier->customer, 1 = customer->supplier
        edge_sources = np.where(edges_df['supplier_or_customer_flag'] == 0, edges_df['partner_idx'], edges_df['gvkey_idx'])
        edge_targets = np.where(edges_df['supplier_or_customer_flag'] == 0, edges_df['gvkey_idx'], edges_df['partner_idx'])

        edge_index = torch.tensor(np.vstack([edge_sources, edge_targets]), dtype=torch.long)

        # Optional: include edge weights
        if edge_weight_col and edge_weight_col in group.columns:
            edge_weights_df = group[['gvkey', 'supplier_or_customer_gvkey', edge_weight_col]].drop_duplicates()
            edge_weights_df['gvkey_idx'] = edge_weights_df['gvkey'].map(global_node_map)
            edge_weights_df['partner_idx'] = edge_weights_df['supplier_or_customer_gvkey'].map(global_node_map)

            # Same direction logic
            edge_weights_df = edge_weights_df.dropna(subset=['gvkey_idx', 'partner_idx'])
            edge_weights_df['source'] = np.where(edge_weights_df['supplier_or_customer_flag'] == 0,
                                                 edge_weights_df['partner_idx'],
                                                 edge_weights_df['gvkey_idx'])
            edge_weights_df['target'] = np.where(edge_weights_df['supplier_or_customer_flag'] == 0,
                                                 edge_weights_df['gvkey_idx'],
                                                 edge_weights_df['partner_idx'])

            # Merge weight info on (source, target)
            weight_map = {(s, t): w for s, t, w in zip(edge_weights_df['source'], edge_weights_df['target'], edge_weights_df[edge_weight_col])}
            edge_attr = torch.tensor([weight_map.get((s.item(), t.item()), 1.0) for s, t in zip(edge_sources, edge_targets)], dtype=torch.float).unsqueeze(1)
        else:
            edge_attr = torch.ones((edge_index.shape[1], 1), dtype=torch.float)  # default weight = 1

        group = group.drop_duplicates(subset='gvkey')
        for _, row in group.iterrows():
            gvkey_idx = global_node_map[row['gvkey']]
            x[gvkey_idx, :] = np.array([row[col] for col in feature_cols], dtype=np.float32)
            if pd.notna(row[response_col]):
                y[gvkey_idx] = row[response_col]

        graphs_per_month[time] = Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,  # <- added
            y=y
        )

    return graphs_per_month, global_node_map