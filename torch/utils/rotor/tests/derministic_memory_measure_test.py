import torch
from rotor import rotor
import os.path


def test_mem_measure():
    print("Starting Test_mem_measure")

    length = 20
    kernel_size = 3

    xbar = x = tmpFwd = tmpBwd = None

    for it in range(0,10):
        module = torch.nn.Sequential()

        for idx in range((length - 1) // (kernel_size - 1)):
            module.add_module(str(idx), torch.nn.Conv1d(1, 1, kernel_size))

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        module.to(device=device)
        net_check = rotor.Checkpointable(module)
        test_path = os.path.abspath(os.path.dirname(__file__))
        data_path = os.path.join(test_path, 'test_data/data_test_conv.pt')
        sample = torch.load(data_path, map_location=torch.device(device))
        net_check.measure(sample)
        result_fwdTime, result_bwdTime, result_xbar, result_x, result_tmpFwd, result_tmpBwd = net_check.all_values
        if it > 1:
            try :
                assert result_xbar == xbar
            except:
                print(it,"x_bar",result_xbar, xbar)
            try:
                assert result_x == x
            except:
                print(it,"x_result",result_x, x)
            try:
                assert result_tmpFwd == tmpFwd
            except:
                print(it,"tmpFWD",result_tmpFwd, tmpFwd)
            try:
                assert result_tmpBwd == tmpBwd
            except:
                print(it,"tmpBWD",result_tmpBwd, tmpBwd)

        xbar = result_xbar
        x = result_x
        tmpFwd = result_tmpFwd
        tmpBwd = result_tmpBwd

    print('test ended')

if __name__ == '__main__':
    test_mem_measure()