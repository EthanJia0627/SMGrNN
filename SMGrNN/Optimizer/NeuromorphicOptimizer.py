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
        self.edge_weight_previous = edge_weight.clone()
        ## =========================  Parameters  =========================
        self.activity_threshold = 0.75         
        ## =========================  Data  =========================
        self.node_activity_history = []        # history of node activity（Tensor）
        self.edge_derivative_history = []        # history of edge weight derivative（Tensor）
        self.timesteps = 0                     # 当前累计的步数
        self.history_window = 100              # 观测时间窗口大小

    def update_state(self):
        """
        Update node activity and store historical info for future growth decision
        """
        # 当前节点活动：特征求和
        self.node_activity = self.g.nodes.sum(dim=1)

        # 记录历史节点活动（detach 避免占用计算图内存）
        self.node_activity_history.append(self.node_activity.detach().clone())
        self.timesteps += 1

        # === 计算 edge 权重变化量（非 Hebbian，而是追踪变化）
        edge_diff = (self.edge_weight - self.edge_weight_previous).detach().clone()
        self.edge_weight_previous = self.edge_weight.clone()

        self.edge_derivative_history.append(edge_diff)
        if len(self.edge_derivative_history) > self.history_window:
            self.node_activity_history.pop(0)
            self.edge_derivative_history.pop(0)
    
    def grow(self):
        """
        Grow the network based on the historical information
        """
        # Check if the network is ready to grow
        if self.timesteps < self.history_window:
            return
        # Calculate the average node activity
        avg_activity = torch.stack(torch.abs(self.node_activity_history)).mean(dim=0)
        # Add new nodes from active nodes
        active_nodes = avg_activity > torch.max(avg_activity.mean()+2*avg_activity.std(),self.activity_threshold)
        ...
        

        
