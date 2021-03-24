import pytest
import torch
from torch.utils.rotor import rotor


def test_get_expected_memory_False():
    with pytest.raises(ValueError):
        print("Starting Test_no-vision")

        length = 5
        kernel_size = 3

        module = torch.nn.Sequential()

        for idx in range((length - 1) // (kernel_size - 1)):
            module.add_module(str(idx), torch.nn.Conv1d(1, 1, kernel_size))

        net_check = rotor.Checkpointable(module)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        module.to(device=device)

        shape = (20, 1, length)
        sample = torch.rand(*shape, device=device)
        net_check.measure(sample)
        net_check.compute_sequence(custom_mem_limit=600)
