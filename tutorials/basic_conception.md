# Activation-based Representation
`spikingjelly.activation_based` uses tensors whose element is only 0 or 1 to represent spikes.

# Data Format
In `spikingjelly.activation_based`, There are two formats of data:
- Data in a single time-step with `shape = [N, *]`, where N is the batch dimension, * represents any extra dimensions.
- Data in many time-steps with `shape = [T, N, *]`, where T is the time-step dimension, N is the batch dimension and * represents any additional dimensions.

# Step Mode
Modules in `spikingjelly.activataion_based` have two propagation modes, which are the shingle-step mode 's' and the multi-step mode 'm'. In single-step mode, the data use the `shape = [N, *] format. In multi-step mode, the data use the `shape = [T, N, *]` format.


The user can set `step_mode` of a module in its `__init__` or change `step_mode` anytime after the module is built.

```python
import torch
from spikingjelly.activation_based import neuron

net = neuron.IFNode(step_mode='m')
# 'm' is the multi-step mode
net.step_mode = 's'
# 's' is the single-step mode
```

If we want to input the sequence data with `shape = [T, N, *]` to a single-step module, we need to implement a for-loop in time-steps manulally, which splits the sequence data into T data with `shape = [N, *]` and sends the data setp-by-step. Let's create a new layer of IF neurons, set it to single-step mode, and input sequence data step-by-step.

```python
import torch
from spikingjelly.activation_based import neuron

net_s = neuron.IFNode(step_mode='s')
T = 4
N = 1
C = 3
H = 8
W = 8
x_seq = torch.rand([T, N, C, H, W])
y_seq = []
for t in range(T):
    x = x_seq[t]  # x.shape = [N, C, H, W]
    y = net_s(x)  # y.shape = [N, C, H, W]
    y_seq.append(y.unsqueeze(0))

y_seq = torch.cat(y_seq)
# y_seq.shape = [T, N, C, H, W]
```

`multi_step_forward` wraps the for-loop in time-steps for single-step modules to handle sequence data with `shape = [T, N, *]`, which is more convenient to use.

```python
import torch
from spikingjelly.activation_based import neuron, functional
net_s = neuron.IFNode(step_mode='s')
T = 4
N = 1
C = 3
H = 8
W = 8
x_seq = torch.rand([T, N, C, H, W])
y_seq = functional.multi_step_forward(x_seq, net_s)
# y_seq.shape = [T, N, C, H, W]
```

However, the best usage is to set the module as a multi-step module directly.
```python
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
```