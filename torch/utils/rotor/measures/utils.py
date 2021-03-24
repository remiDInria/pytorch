import warnings
import torch

__all__ = ["RngState"]

def tensor_mem_size(tensors):
    """
    Recursively sum the memory sizes of the torch.tensor contained in tensors.
    :param tensors: torch.Tensor or container of torch.tensor.
    :return: memory size of the tensor in bytes.
    """
    if isinstance(tensors, torch.Tensor):
        return tensors.element_size() * tensors.nelement()
    else:
        return sum(tensor_mem_size(u) for u in tensors)


def make_gradient_for(tensors):
    """
    Builds torch.ones tensors according to the shape of the tensors contained in tensors.
    :param tensors: torch.Tensor or container of torch.tensor.
    :return: tuple(torch.Tensor)
    """
    if isinstance(tensors, torch.Tensor):
        return torch.ones_like(tensors)
    else:
        return tuple(torch.ones_like(u) for u in tensors)


# Allows a module that inherits from Sequential
# to specify that it overrides the forward() method
# and thus should be considered as a single block
def is_really_sequential(m):
    b = isinstance(m, torch.nn.Sequential)
    if b:
        try:
            if m.not_really_sequential:
                return False
            return True
        except AttributeError:
            return True
    else:
        return False


# Extracts a sequence of modules from a given module, recursively
# Returns a list of (name, module) where running the modules in sequence
# is equivalent to running the input module
# Names are formatted with the same format as onnx does.
def extract_children_from_sequential(m, name="", prefix=""):
    class_name = type(m).__name__
    full_name = prefix + ("/" if prefix else "")
    full_name += class_name
    if name:
        full_name += "[" + name + "]"
    if not (is_really_sequential(m)):
        return [(full_name, m)]
    children = (extract_children_from_sequential(c, name=n, prefix=full_name) for (n, c) in m.named_children())
    children = sum(children, [])
    return children


def detach_variable(inputs, force_required_grad=False):
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = force_required_grad or inp.requires_grad
            out.append(x)
        return tuple(out)

    elif isinstance(inputs, torch.Tensor):
        out = inputs.detach()
        out.requires_grad = force_required_grad or inputs.requires_grad
        return out
    else:
        raise RuntimeError(
            "Only Tensor or tuple of Tensors is supported. Got Unsupported input type: ", type(inputs).__name__)


def check_backward_validity(inputs):
    if not any(inp.requires_grad for inp in inputs):
        warnings.warn("None of the inputs have requires_grad=True. Gradients will be None")


def ensure_tuple(output):
    r"""
    This function builds a tuple if the the object given as argument is a torch Tensor.
    :param output: Object to build the tuple with, (python tuple or torch Tensor).
    :type output: Any
    :return: Python tuple build from input Tensor.
    :rtype: Python Tuple
    """
    if isinstance(output, torch.Tensor):
        return (output,)
    return output


def get_gradients(inputs):
    if isinstance(inputs, torch.Tensor):
        return inputs.grad
    else:
        return tuple(inp.grad if isinstance(inp, torch.Tensor) else inp
                     for inp in inputs)


class RngState:
    counter = 0

    def __init__(self, tensors):
        self.counter = RngState.counter
        RngState.counter += 1
        self.cpu_state = torch.get_rng_state()
        self.had_cuda = False
        if torch.cuda._initialized:
            self.had_cuda = True
            self.gpu_devices = list(set(t.get_device() for t in tensors
                                        if isinstance(t, torch.Tensor) and t.is_cuda))
            self.gpu_states = []
            for device in self.gpu_devices:
                with torch.cuda.device(device):
                    self.gpu_states.append(torch.cuda.get_rng_state())

    def restore(self):
        torch.set_rng_state(self.cpu_state)
        if self.had_cuda:
            for device, state in zip(self.gpu_devices, self.gpu_states):
                with torch.cuda.device(device):
                    torch.cuda.set_rng_state(state)
