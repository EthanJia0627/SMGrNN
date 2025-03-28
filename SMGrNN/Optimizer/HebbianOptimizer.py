import torch
from ..Graph.DirectedGraph import DirectedGraph
from torch.nn import Parameter

class HebbianOptimizer:
    def __init__(self,graph:DirectedGraph,edge_weight:Parameter):
        ## =========================  Graph  =========================
        self.g = graph
        self.edge_weight = edge_weight
        ## =========================  Parameters  =========================
        ...
        ## =========================  Data  =========================
        self.node_activity_history = []        # history of node activity（Tensor）
        self.edge_collaborativity_history = []        # history of edge weight derivative（Tensor）
        self.timesteps = 0                     # 当前累计的步数
        self.history_window = 100              # 观测时间窗口大小

    def update_state(self):
        self.node_activity = self.g.nodes.sum(dim=1)
        edge_deriv = self.node_activity.unsqueeze(1) @ self.node_activity.unsqueeze(0)
        self.edge_weight_derivative = edge_deriv

        # === 累计历史 ===
        self.node_activity_history.append(self.node_activity.detach().clone())
        self.edge_collaborativity_history.append(edge_deriv.detach().clone())
        self.timesteps += 1

        # 限制窗口长度
        if len(self.node_activity_history) > self.history_window:
            self.node_activity_history.pop(0)
            self.edge_collaborativity_history.pop(0)
    
    def step(self):
        ...