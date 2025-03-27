import torch
from ..Graph.DirectedGraph import DirectedGraph
from torch.nn import Parameter

class HebbianOptimizer:
    def __init__(self,graph:DirectedGraph,edge_weight:Parameter):
        self.g = graph
        self.edge_weight = edge_weight
    