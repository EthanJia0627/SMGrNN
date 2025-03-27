import torch
from ..Graph.DirectedGraph import DirectedGraph
from torch.nn import Parameter

class NeuromorphicOptimizer:
    """
    Neuromorphic Growth Optimizer for the network based on node activity and edge weight
    """
    def __init__(self,graph:DirectedGraph,edge_weight:Parameter):
        ## =========================  Graph  =========================
        self.g = graph
        self.edge_weight = edge_weight
        ## =========================  Parameters  =========================
        ...
        ## =========================  Data  =========================
        self.node_activity = torch.zeros(self.g.nodes.size(0),device=self.g.nodes.device)
        self.edge_weight_derivative = torch.zeros(self.edge_weight.size(),device=self.edge_weight.device)

    

        