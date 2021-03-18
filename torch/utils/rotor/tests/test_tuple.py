import torch
torch.utils.rotor import rotor

r""" Test tuples as input to rotor module instead of tensor.
"""
def test_tuple():
    shape = (1,1,100)
    length = 18
    module = torch.nn.Sequential()
    module.add_module("fork", ForkConv())
    for idx in range(length - 2):
        module.add_module(str(idx), BiConv())
    module.add_module("join", Join())

    for (n, p) in module.named_parameters():
        p.grad = torch.zeros_like(p)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    data = torch.rand(*shape, device=device)
    data.requires_grad = True

    module.to(device)
    net_check = rotor.Checkpointable(module)
    net_check.measure(data)
    net_check.compute_sequence(custom_mem_limit=700 * 1024 * 1024)
    result = net_check(data).sum()
    result.backward()
    data.grad

class ForkConv(torch.nn.Module):
    def __init__(self):
        super(ForkConv, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 1, 3)
        self.conv2 = torch.nn.Conv1d(1, 1, 3)

    def forward(self, x):
        a = self.conv1(x)
        b = self.conv2(x)
        return (a, b)


class BiConv(torch.nn.Module):
    def __init__(self):
        super(BiConv, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 1, 3)
        self.conv2 = torch.nn.Conv1d(1, 1, 3)

    def forward(self, xs):
        a = self.conv1(xs[0])
        b = self.conv2(xs[1])
        return (a, b)

class Join(torch.nn.Module):
    def __init__(self):
        super(Join, self).__init__()

    def forward(self, xs):
        return xs[0] + xs[1]

