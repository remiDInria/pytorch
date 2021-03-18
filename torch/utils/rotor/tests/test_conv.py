import torch
from torch.utils.rotor import rotor
import os.path

def test_backward():
    print("Starting Test_no-vision")

    length = 70
    kernel_size = 3

    module = torch.nn.Sequential()

    for idx in range((length - 1) // (kernel_size - 1)):
        module.add_module(str(idx), torch.nn.Conv1d(1, 1, kernel_size))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    module.to(device=device)
    net_check = rotor.Checkpointable(module)

    #test_path = os.path.abspath(os.path.dirname(__file__))
    #"data_path = os.path.join(test_path, 'test_data/data_test_conv.pt')
    #"sample = torch.load(data_path, map_location=torch.device(device))

    shape = (20, 1, length)
    sample = torch.rand(*shape, device=device)

    net_check.measure(sample)
    mem_limit = 550000
    net_check.compute_sequence(custom_mem_limit=mem_limit)
    #net_check.compute_sequence(custom_mem_limit=1500000)
    #net_check.compute_sequence(custom_mem_limit=10000000)

    #data = torch.load(data_path, map_location=torch.device(device))
    data = torch.rand(*shape, device=device)
    data.requires_grad = True
    result = net_check(data).sum()
    result.backward()
    data.grad

if __name__ == '__main__':
    test_backward()
