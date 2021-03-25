import torch
import pytest
import torch.nn as nn
from torch.utils.rotor import rotor
from torch.utils.rotor.measures.utils import *

from torch.utils.rotor.tests.test_resnet import measure_consistency

class AtomicSequential(nn.Sequential):
        def __init__(self, *args, **kwargs):
            super(AtomicSequential, self).__init__(*args, **kwargs)
            self.not_really_sequential = True

## Because in-place operations do not mix well with partial checkpointing
## This class performs an in-place ReLU operation after
## the given module

class ReLUatEnd(nn.Module):
    def __init__(self, module):
        super(ReLUatEnd, self).__init__()
        self.module = module
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.module(x)
        x = self.relu(x)
        return x

def test_cuda_preserve():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    do_test()

def test_cuda_no_preserve():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    with pytest.raises(AssertionError):
        do_test(preserve_rng=False)

def do_test(length=25, device='cuda', preserve_rng=True):
    device = torch.device(device)

    net = torch.nn.Sequential()
    net.add_module("start", torch.nn.Conv2d(3, 10, kernel_size=3, padding=1))
    for idx in range(length):
        if idx % 2 == 0:
            net.add_module(str(idx), AtomicSequential(torch.nn.Conv2d(10, 10, kernel_size=3, padding=1),
                                                      torch.nn.Dropout(p=0.5, inplace=True)))
        else:
            net.add_module(str(idx), ReLUatEnd(torch.nn.Conv2d(10, 10, kernel_size=3, padding=1)))
    limit = 600

    net.to(device=device)
    original_state = net.state_dict()
    for (k, v) in original_state.items():
        original_state[k] = v.clone()
    shape = (32, 3, 224, 224)

    net_check = rotor.Checkpointable(net, preserve_rng_state=preserve_rng)
    sample = torch.rand(*shape, device=device)
    net_check.measure(sample)
    net_check.compute_sequence(custom_mem_limit=limit * 1024 * 1024)
    net.load_state_dict(original_state)

    print(net_check.chain)
    print(net_check.sequence)

    data = torch.rand(*shape, device=device).requires_grad_()

    ambient_error = max(1.25 * measure_consistency(net, data), 1e-6)

    data_clone = data.clone().detach().requires_grad_()
    rng_state = RngState(data)
    result = net_check(data).sum()
    result.backward()

    net.load_state_dict(original_state)
    rng_state.restore()
    result_clone = net(data_clone).sum()
    result_clone.backward()

    assert(torch.allclose(result, result_clone))
    diff = torch.abs(data.grad - data_clone.grad)
    print("Distance between grads", torch.max(diff).item())
    assert(torch.allclose(data.grad, data_clone.grad, atol=ambient_error))
