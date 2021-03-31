import torch
from torch.utils.rotor import memory
from torch.utils.rotor import timing
from torch.utils.rotor import utils
import math

__all__ = ["Chain"]

class Chain:
    r"""
    Holds forward and backward sequence memory and duration discretized measures.
    Memory measures are discretized in slots of mem_units bytes

    :param forward_durations: forward durations in ms.
    :type forward_durations: list[float]
    :param backward_durations: backward durations in ms.
    :type : list[float]
    :param forward_memory_sizes: forward input memory consumption in Checkpointable.mem_units. # activation_size
    :type forward_memory_sizes: list[int]
    :param backward_memory_sizes: backward input memory consumption in Checkpointable.mem_units. #
    :type backward_memory_sizes: list[int]
    :param forward_memory_tmp_sizes: additional memory consumption during forward phase in Checkpointable.mem_units.
    :type  forward_memory_tmp_sizes: list[int]
    :param backward_memory_temp_sizes: additional memory consumption during backward phase in Checkpointable.mem_units.
    :type backward_memory_temp_sizes: list[int]
    :param length: length of the Checkpointable sequence.
    :tye length: int
    """
    # TODO(REMI) exportable / serialisable  pickle?
    def __init__(self, all_values, mem_unit, check=True, loss_tmp_memory_usage=0):
        r"""
        Constructor which discretizes memory values and expresses them 
        in slots of mem_units.

        :param all_values: values as returned from measure_everything
        :type all_values: (list[float], list[float], list[int], list[int], list[int], list[int])
        :param mem_unit: discretization parameter
        :type mem_unit: int
        :param check: Performs a validation of the consistence of the list parameters length. Default True.
        :type check: bool
        :param loss_tmp_memory_usage: Memory usage of the loss operation
        :type loss_tmp_memory_usage: int
        """

        def discretize(values):
            return [math.ceil(value / mem_unit) for value in values]

        if len(all_values) != 6:
            raise AttributeError("In Chain, input all_values has length %d instead of 6" % len(all_values))
        self.forward_durations = all_values[0]
        self.backward_durations = all_values[1] + [0]
        self.activation_total_usages = discretize(all_values[2])
        self.activation_sizes = discretize(all_values[3])
        self.forward_memory_tmp_sizes = discretize(all_values[4])
        self.backward_memory_tmp_sizes = discretize(all_values[5] + [loss_tmp_memory_usage])
        self.length = len(self.forward_durations)
        if check and not self.check_lengths():
            raise AttributeError("In Chain, input lists do not have consistent lengths")

    def check_lengths(self):
        r""" Checks the consistence of the length of list attributes.

        :return:
        :rtype: bool
        """
        return ((len(self.forward_durations) == self.length)
                and (len(self.backward_durations) == self.length + 1)
                and (len(self.activation_total_usages) == self.length + 1)
                and (len(self.activation_sizes) == self.length + 1)
                and (len(self.forward_memory_tmp_sizes) == self.length)
                and (len(self.backward_memory_tmp_sizes) == self.length + 1))

    def __repr__(self):
        line = []
        for i in range(self.length):
            line.append(
                (self.forward_durations[i], self.backward_durations[i], self.activation_sizes[i],
                 self.activation_total_usages[i], self.forward_memory_tmp_sizes[i], self.backward_memory_tmp_sizes[i]))
        i = self.length
        line.append((None, self.backward_durations[i], self.activation_sizes[i],
                     self.activation_total_usages[i], None, self.backward_memory_tmp_sizes[i]))
        return line.__repr__()



# Measure execution time and memory usage
# just by running each block in sequence
# Durations are in ms, memory sizes are in bytes
def measure_everything(named_modules, custom_input, min_duration_ms=30):

    # Builds zero gradient for each tensor of the model.
    for (_, m) in named_modules:
        for p in m.parameters():
            p.grad = torch.zeros_like(p)

    current_input_tensor = utils.detach_variable(custom_input)
    current_input_tensor.requires_grad = True
    fwd_durations_ms = []  # Forward operations durations in ms.
    bwd_durations_ms = []  # Backward durations in ms.
    activation_total_usages = [utils.tensor_mem_size(custom_input)]  # Memory occupied by the result of each forward operation. This includes the tensors in the computational graph built by the forward operation
    activation_sizes = [utils.tensor_mem_size(custom_input)]  # Memory size of the activation returned by each forward operation, not counting the computational graph.
    fw_tmp_mem_usages = []  # Additional temporary use of memory during forward operations.
    bw_tmp_mem_usages = []  # Additional temporary use of memory during backward operations.

    # GPU Initialization
    with torch.enable_grad():
        y = named_modules[0][1](current_input_tensor)
    torch.autograd.backward(y, grad_tensors=utils.make_gradient_for(y))
    del y

    timer = timing.make_timer(custom_input.device)
    mem_measurer = memory.MeasureMemory(custom_input.device)

    def perform_measure(func, prologue=None):
        """
        Calls memory.MeasureMemory on func.
        Applies prologue function first if defined.
        Performs timing duration measures of func execution.
        This method is called for forward operations only.
        :param func: the function to be measured.
        :param prologue: Prologue function (optional)
        :return: function execution duration in ms, memory usage in memory maximum usage in bytes.
        """
        def complete_func():
            if prologue:
                prologue()
            return func()

        _, complete_func_mem_usage, complete_func_max_mem_usage = mem_measurer.measure(complete_func)
        func_duration_ms = timer.measure_median(func)
        if func_duration_ms < min_duration_ms:
            number_repetitions = 1 + int(min_duration_ms // func_duration_ms)
            func_duration_ms = timer.measure_median(func, iterations=number_repetitions)
        return func_duration_ms, int(complete_func_mem_usage), int(complete_func_max_mem_usage)

    for name, module in named_modules:
        current_input_tensor = utils.detach_variable(current_input_tensor)

        def forward_operation():
            nonlocal fwd_result
            fwd_result = None
            with torch.enable_grad():
                fwd_result = module(current_input_tensor)

        fwd_duration_ms, fwd_mem_usage, fwd_max_mem_usage = perform_measure(forward_operation)

        activation_size = utils.tensor_mem_size(fwd_result)
        activation_sizes.append(activation_size)
        activation_usage = max(fwd_mem_usage, activation_size)
        activation_total_usages.append(activation_usage)
        fwd_durations_ms.append(fwd_duration_ms)

        # with torch.enable_grad():
        #     summary = xbar.sum()
        # xbar = None

        def backward_operation():
            if isinstance(current_input_tensor, torch.Tensor):
                current_input_tensor.grad = None
            args = utils.make_gradient_for(fwd_result)
            torch.autograd.backward(fwd_result, grad_tensors=args)
            # summary.backward()

        # memDisplay.printCurrentState("Measuring Bwd" + name)
        # Measure backward only once, because precise timings are not needed since all
        # backwards are only performed once, no optimization available here
        # Plus running bwd several times would require retain_graph=True, and
        # it might modify the memory usage
        bwd_duration, _, bwd_max_mem_usage = mem_measurer.measure(lambda: timer.measure(backward_operation))
        bwd_durations_ms.append(bwd_duration)

        with torch.no_grad():
            fwd_result = module(current_input_tensor)

        fw_tmp_mem_usages.append(int(fwd_max_mem_usage) - activation_usage) # input was already in memory when starting experiment
        bw_tmp_mem_usages.append(int(bwd_max_mem_usage) - (utils.tensor_mem_size(current_input_tensor) +
                                                           utils.tensor_mem_size(fwd_result))) # input x_i and xb_i+1 were in memory, y_i+1 and y_i were added.

        current_input_tensor = utils.detach_variable(fwd_result, force_required_grad=True)
        del fwd_result

    for (_, m) in named_modules:
        m.zero_grad()

    return fwd_durations_ms, bwd_durations_ms, activation_total_usages, activation_sizes, fw_tmp_mem_usages, bw_tmp_mem_usages
