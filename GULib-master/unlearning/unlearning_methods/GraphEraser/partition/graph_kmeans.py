from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from collections import defaultdict
from sklearn.cluster import KMeans
import torch
class PartitionGraphKM():
    def __init__(self, args,logger, data):
        self.args = args
        self.logger = logger
        self.data = data[0]

    def partition(self):
        train_loader = DataLoader(self.data, batch_size=64, shuffle=False)
        all_x = []
        community_to_graph = defaultdict(list)
        for data in train_loader:
            x = global_mean_pool(data.x, data.batch)
            all_x.append(x)
            
        graph_embedding = torch.cat(all_x, dim=0)
        n_clusters = self.args["num_shards"]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(graph_embedding)
        
        for node_id, community_id in enumerate(labels):
            community_to_graph[community_id].append(node_id)
            
        community_to_graph = dict(community_to_graph)

        return community_to_graph
                
    