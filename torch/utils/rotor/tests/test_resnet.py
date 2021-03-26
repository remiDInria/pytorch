import torch
import torch.nn as nn
from torch.utils.rotor import rotor
from torch.utils.rotor.algorithms.sequence import *
from torch.utils.rotor.measures.memory import *
from torch.utils.rotor.measures.utils import *
import pytest
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_device_type import onlyCUDA, instantiate_device_type_tests
import pdb

class ResNet(nn.Sequential):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        super(ResNet, self).__init__()
        self.add_module('conv1', nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                           bias=False))
        self.add_module('bn1', ReLUatEnd(norm_layer(self.inplanes)))
        self.add_module('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.add_module('layer1', self._make_layer(block, 64, layers[0]))
        self.add_module('layer2', self._make_layer(block, 128, layers[1], stride=2,
                                                   dilate=replace_stride_with_dilation[0]))
        self.add_module('layer3', self._make_layer(block, 256, layers[2], stride=2,
                                                   dilate=replace_stride_with_dilation[1]))
        self.add_module('layer4', self._make_layer(block, 512, layers[3], stride=2,
                                                   dilate=replace_stride_with_dilation[2]))
        self.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
        self.add_module('flatten', Flatten())
        self.add_module('fc', nn.Linear(512 * block.expansion, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


# Because in-place operations do not mix well with partial checkpointing
# This class performs an in-place ReLU operation after
# the given module

class ReLUatEnd(nn.Module):
    def __init__(self, module):
        super(ReLUatEnd, self).__init__()
        self.module = module
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.module(x)
        x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)

def strToSeq(str):
    words = str.split(', ')
    result = Sequence(Function("FromStr"))
    for w in words:
        if w == "L":
            result.insert(Loss())
            continue
        op, index = w.split('_')
        index = int(index)
        if op == "Fn":
            op = ForwardNograd(index)
        elif op == "Fe":
            op = ForwardEnable(index)
        elif op == "CF":
            op = ForwardCheck(index)
        elif op == "B":
            op = Backward(index)
        else:
            raise ArgumentError("Wrong operation " + w)
        result.insert(op)
    return result


class TestResnet(TestCase):

    def do_test(self, depth=18, device='cuda', hardcoded_sequence=None, use_limit=True, verbosity=0):

        device = torch.device(device)
        if depth == 18:
            net = resnet18()
            limit = 560
        else:
            net = resnet152()
            limit = 2048
        net.to(device=device)
        original_state = net.state_dict()
        for (k, v) in original_state.items():
            original_state[k] = v.clone()
        shape = (32, 3, 224, 224)

        net_check = rotor.Checkpointable(net, verbosity=verbosity)
        if hardcoded_sequence:
            net_check.sequence = hardcoded_sequence
            use_limit=False
        else:
            sample = torch.rand(*shape, device=device)
            net_check.measure(sample)
            if use_limit: net_check.compute_sequence(custom_mem_limit=limit * 1024 * 1024)
            else: net_check.compute_sequence()
            net.load_state_dict(original_state)

        if verbosity:
            display = DisplayMemory(device)
            display.inspectModule(net)

        print(net_check.chain)
        print(net_check.sequence)


        measurer = MeasureMemory(device)
        mem_usage_before = measurer.current_mem_usage()

        data = torch.rand(*shape, device=device).requires_grad_()
        #pdb.set_trace()
        ambient_error = max(1.25 * measure_consistency(net, data), 1e-6)

        measurer.reset_max_memory()
        result = net_check(data).sum()
        result.backward()
        mem_usage_during = measurer.maximum_value()

        if device.type == 'cuda' and use_limit:
            print(mem_usage_during, mem_usage_before, limit)
            assert(int(mem_usage_during)  - int(mem_usage_before) <= limit * 1024 * 1024)

        data_clone = data.clone().detach().requires_grad_()
        net.load_state_dict(original_state)
        result_clone = net(data_clone).sum()
        result_clone.backward()

        assert(torch.allclose(result, result_clone))
        diff = torch.abs(data.grad - data_clone.grad)
        print("Distance between grads", torch.max(diff).item())
        assert(torch.allclose(data.grad, data_clone.grad, atol=ambient_error))

    def test_resnet18_cpu(self, device):
        print("test_resnet18_cpu")
        self.do_test(device=device)

    def test_resnet18_cpu_hc(self, device):
        print("test_resnet18_cpu_hc")
        sequence = strToSeq("CF_0, Fn_1, Fn_2, Fe_3, Fe_4, Fe_5, Fe_6, Fe_7, Fe_8, Fe_9, Fe_10, Fe_11, Fe_12, Fe_13, L, B_13, B_12, B_11, B_10, B_9, B_8, B_7, B_6, B_5, B_4, B_3, Fe_0, Fe_1, Fe_2, B_2, B_1, B_0")
        self.do_test(device=device, hardcoded_sequence=sequence)

    @onlyCUDA
    def test_resnet18_cuda(self, device):
        self.do_test(device=device)

    @onlyCUDA
    def test_resnet152_cuda(self,device):
        self.do_test(depth=152, device=device)

    def test_resnet18_cpu_verbose(self, device):
        self.do_test(device=device, verbosity=6)

    @onlyCUDA
    def test_resnet18_cuda_verbose(self, device):
        self.do_test(device=device, verbosity=6)

    def test_resnet18_cpu_nolimit(self, device):
        self.do_test(device=device, use_limit=False)

    @onlyCUDA
    def test_resnet18_cuda_nolimit(self, device):
        self.do_test(device=device, use_limit=False)


instantiate_device_type_tests(TestResnet, globals())

def measure_consistency(network, data, save_state=True):
    data = data.clone().detach().requires_grad_()
    if save_state:
        original_state = network.state_dict()
        #pdb.set_trace()
        for (k, v) in original_state.items():
            original_state[k] = v.clone()
    data_clone = data.clone().detach().requires_grad_()
    rng_state = RngState(data)
    result = network(data).sum()
    result.backward()
    original_grad = data.grad

    if save_state:
        network.load_state_dict(original_state)
    rng_state.restore()
    result_clone = network(data_clone).sum()
    result_clone.backward()
    
    assert(torch.allclose(result, result_clone))
    diff = torch.abs(original_grad - data_clone.grad)
    return torch.max(diff).item()


if __name__ == '__main__':
    run_tests()

#if __name__ == "__main__":
#    do_test(device='cpu')


