import torch
from spikingjelly.activation_based import neuron

net_m = neuron.IFNode(step_mode='m')
T = 4
N = 1
C = 3
H = 8
W = 8
x_seq = torch.rand([T, N, C, H, W])
y_seq = net_m(x_seq)
# y_seq.shape = [T, N, C, H, W]