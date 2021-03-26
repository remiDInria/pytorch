import sys
import torch

from torch.utils.rotor import measures
from torch.utils.rotor import algorithms as alg
from torch.utils.rotor.algorithms import sequence as sq

__all__ = ["TensorStorage", "CheckpointOptim", "Checkpointable"]



class TensorStorage:
    def __init__(self):
        self.storage = {}  # storage[i] stores the input of functions[i]
        self.sourceStorage = {}
        # if storage[i] has a computation graph,
        # sourceStorage[i] is the input to the
        # call that created it
        self.rngStorage = {}

    def store_value(self, index, val, source, rng_state=None):
        self.storage[index] = val
        self.sourceStorage[index] = source
        self.rngStorage[index] = rng_state

    def get_value(self, index):
        return self.storage[index]

    def get_source(self, index):
        return self.sourceStorage[index]

    def get_rng(self, index):
        return self.rngStorage[index]

    def delete_index(self, index):
        del self.storage[index]
        del self.sourceStorage[index]
        del self.rngStorage[index]

    def __str__(self, *args):
        def key_to_str(key):
            suffix = ""
            if self.sourceStorage[key] is not None:
                suffix += "*"
            if self.rngStorage[key] is not None:
                suffix += "^"
            return str(key) + suffix

        key_list = " ".join(key_to_str(k) for k in self.storage.keys())
        return "Size {}, keys {}".format(len(self.storage), key_list)

    # These methods are useful to be able to use save_for_backward()
    # for all the tensors that are kept between the forward and backward
    # phases. For the moment it is a noop (see commented sections below) 
    # because it does not work properly for our case. The missing feature
    # is the possibility to perform 
    # 
    # y = f(x)
    # save_for_backward(x, y)
    # 
    # in the forward phase, and then 
    # 
    # x, y = saved_tensors
    # y.backward(grad)
    # grad = x.grad
    #
    # in the backward phase. As of now, this results in `x.grad` being
    # `None` instead of containing useful data (see $580)
    # 
    # If this feature can not be implemented, this restricts the set of possible
    # checkpointing stategies. It is actually still possible to compute the
    # optimal sequence in this restricted set, so we would (only) have to adapt
    # our algorithms. 

    def serialize(self):
        self.result = tuple()

        def save(tensors):
            if tensors is None:
                return None
            tensors = measures.utils.ensure_tuple(tensors)
            start_index = len(self.result)
            self.result = self.result + tensors
            end_index = len(self.result)
            return start_index, end_index

        for k in self.storage.keys():
            pass
            # self.storage[k] = save(self.storage[k])
            # self.sourceStorage[k] = save(self.sourceStorage[k])
        result = self.result
        del self.result
        return result

    @staticmethod
    def _deserialize_helper(dictionary, saved_tensors):
        for k in dictionary.keys():
            if dictionary[k] is not None:
                start, end = dictionary[k]
                if end == start + 1:
                    dictionary[k] = saved_tensors[start]
                else:
                    dictionary[k] = saved_tensors[start:end]

    def deserialize(self, saved_tensors):
        pass
        # self._deserialize_helper(self.storage, saved_tensors)
        # self._deserialize_helper(self.sourceStorage, saved_tensors)


class CheckpointOptim(torch.autograd.Function):
    r"""This computes a sequence of functions, following the sequence of
    operations given as argument. A selected subset of activations are
    stored during the forward phase, some with their computation
    graph, some without. The backward phase follows the end of the
    sequence, with some recomputations when values are missing.
    """

    @staticmethod
    def forward(ctx, functions, sequence, names, preserve_rng_state, arg):
        r""" Overriding unimplemented torch.autograd.Function.forward method.


        :param ctx: pytorch context
        :type ctx: Any
        :param functions: List of sequential function to apply in the sequential model.
        :type functions: list<torch.nn.modules>.
        :param sequence: List of sequential operations to apply to arg.
        :type sequence: List<rotor.algoriths.sequence.Operation>
        :param names: Names of functions.
        :type names: list<string>.
        :param preserve_rng_state:
        :type preserve_rng_state: bool.
        :param arg: input tensor or tuple
        :type arg: torch.Tensor or tuple
        :return:
        :rtype: torch.Tensor or tuple
        """
        measures.utils.check_backward_validity(arg)
        current_input = arg
        ctx.run_function = functions
        ctx.names = names
        ctx.preserve_rng_state = preserve_rng_state
        storage = TensorStorage()
        storage.store_value(0, arg, None)
        source_of_current = None
        loss_operation_index = 0
        for idx, op in enumerate(sequence):
            if names:
                print(op, names[op.index] if hasattr(op, 'index') else "", file=sys.stderr)

            if type(op) is sq.ForwardEnable:
                # A theorem says: ForwardEnable operations are never done twice. So there is no
                # need to save the RNG state here.
                storage.store_value(op.index, current_input, source_of_current)
                current_input = measures.utils.detach_variable(current_input, force_required_grad=True)
                source_of_current = current_input
                with torch.enable_grad():
                    current_input = functions[op.index](current_input)

            elif isinstance(op, sq.Forward):  # covers both ForwardNograd and ForwardCheck
                if type(op) is sq.ForwardCheck:
                    storage.store_value(op.index, current_input, source_of_current,
                                        measures.utils.RngState(current_input) if preserve_rng_state else None)
                with torch.no_grad():
                    current_input = functions[op.index](current_input)
                source_of_current = None

            elif type(op) is sq.Loss:
                loss_operation_index = idx
                break
            elif type(op) is sq.Backward:
                raise ValueError("Encountered Backward op {op} in Forward phase, index {idx}".format(op=op, idx=idx))
            else:
                raise AttributeError("Unknown operation type {t} {op}".format(t=type(op), op=op))

        # Save the last computed value (it is always a ForwardEnable operation)
        if loss_operation_index > 0:
            last_index = sequence[loss_operation_index - 1].index
            storage.store_value(last_index + 1, current_input, source_of_current)
        ctx.sequence = sequence[loss_operation_index + 1:]
        ctx.save_for_backward(*storage.serialize())
        ctx.storage = storage
        if loss_operation_index > 0:
            return measures.utils.detach_variable(current_input)
        else:
            return current_input

    @staticmethod
    def backward(ctx, *args):
        names = ctx.names
        preserve_rng_state = ctx.preserve_rng_state
        sequence = ctx.sequence
        functions = ctx.run_function
        storage = ctx.storage
        storage.deserialize(ctx.saved_tensors)

        idx = 0
        while idx < len(sequence):
            op = sequence[idx]
            if names:
                print(op, names[op.index] if hasattr(op, 'index') else "", "Usage: ", storage, file=sys.stderr)
            if isinstance(op, sq.Forward):
                current_input = storage.get_value(op.index)
                state = storage.get_rng(op.index)
                source = None
                if type(op) is sq.ForwardEnable:
                    current_input = measures.utils.detach_variable(current_input, True)
                    source = current_input
                    if state:
                        storage.rngStorage[op.index] = None  # no longer needed, we will not do this forward again

                if state:
                    state.restore()
                elif type(op) is sq.ForwardCheck and preserve_rng_state:
                    state = measures.utils.RngState(current_input)
                    storage.rngStorage[op.index] = state
                with torch.set_grad_enabled(type(op) is sq.ForwardEnable):
                    current_input = functions[op.index](current_input)
                storage.store_value(op.index + 1, current_input,
                                    source)  # not saving state now, state will be saved if needed just before next Fwd
                if type(op) is sq.ForwardNograd:
                    storage.delete_index(op.index)
                del current_input

            elif type(op) is sq.Loss:
                raise ValueError("Encountered Loss op {op} in Backward phase, index {idx}".format(op=op, idx=idx))

            elif type(op) is sq.Backward:
                src_index = op.index + 1
                torch.autograd.backward(storage.get_value(src_index), grad_tensors=args)
                args = measures.utils.get_gradients(storage.get_source(src_index))
                assert args is not None
                storage.delete_index(src_index)

            idx += 1

        if isinstance(args, torch.Tensor):
            return (None, None, None, None, args)
        else:
            return (None, None, None, None, *args)



class Checkpointable(torch.nn.Module):
    r"""
    Main class of rotor module.
    Holds pytorch sequential module, user params,
    measures and calculated optimal forward sequence.

    ..Input:
    :param model: pytorch sequential module.
    :type model: torch.nn.module.

    ..Input Description:
    :param modules_and_names: list description of the model.
    :type modules_and_names: list[tuple[str, torch.nn.Module]].
    :param names: list of names of the modules contained in model.
    :type names: list[str]
    :param functions: list of the modules contained in model.
    :type functions: list[torch.nn.Module]

    ..User parameters:
    :param verbosity: Verbosity level. TODO:logger
    :type verbosity: int
    :param mem_slots: Quantity of memory buckets.
    Used for discretization in the dynamic sequence optimisation module.
    Default 500.
    :type mem_slots: int
    :param mem_limit: custom memory limit in bytes. Default to None
    :type mem_limit: int
    :param preserve_rng_state: when True (the default), Checkpointable saves the state of the 
    random number generator, and restores it before each recomputation. Should be True
    if your model uses randomized layers
    :type preserve_rng_state: bool

    ..measures:
    :param all_values: List containing memory and time consumptions of a forward/backward pass.
    :type all_values: tuple[list[float]]
    :param chain: Discretized and mem_unit converted all_values memory measures.
    :type chain: rotor.Chain
    :param loss_tmp_memory_usage: Customer parameter. Additional memory used between forward and backward :
    loss function or Checkpointable uncovered  parts of the network.
    Default to 0.
    :type loss_tmp_memory_usage: int.

    ..calculated:
    :param sequence: Optimal calculated forward sequence.
    :type sequence: rotor.algorithms.sequence

    TODO(REMI): reactiver la classe params avec valeurs par défaut?
    TODO(REMI): réduire le nombre d'attributs.
    """

    def __init__(self, model, custom_input=None, mem_limit=None,
                 mem_slots=500, verbosity=0, preserve_rng_state=True, loss_tmp_memory_usage=0):
        r"""
        :param model: pytorch sequential module.
        :type model: torch.nn.module.
        :param custom_input: input data.
        :type custom_input: torch.Tensor
        :param mem_limit: User defined limit memory in bytes.
        :type mem_limit: int
        :param mem_slots: quantity of memory buckets.
        Used for discretization in the dynamic sequence optimisation module.
        Default 500.
        :type mem_slots: int
        :param verbosity: verbosity level
        :type verbosity: int
        :param preserve_rng_state: freeze the random number generator.
        :type preserve_rng_state: bool
        param loss_tmp_memory_usage: Customer parameter. Additional memory used between forward and backward :
        loss function or Checkpointable uncovered parts of the network.
        Default to 0.
        :type loss_tmp_memory_usage: int.
        """
        super(Checkpointable, self).__init__()
        self.model = model
        self.modules_and_names = measures.utils.extract_children_from_sequential(model)
        self.names, self.functions = (list(vals) for vals in zip(*self.modules_and_names))
        self.verbosity = verbosity
        self.mem_slots = mem_slots
        self.preserve_rng_state = preserve_rng_state
        self.all_values = None
        self.chain = None
        self.sequence = None
        self.loss_tmp_memory_usage = loss_tmp_memory_usage
        self.mem_limit = mem_limit

        if custom_input is not None:
            self.measure(custom_input)
        if mem_limit is not None:
            self.compute_sequence(mem_limit)

    def measure(self, custom_input):
        r"""
        Effectuates time and memory usage measures on a forward and simulated backward
        pass sequential of self.model on custom input.

        :param custom_input: input data.
        :type custom_input: torch.nn.Tensor
        """
        self.all_values = measures.inspection.measure_everything(self.modules_and_names, custom_input)
        self.sequence = None

    def build_chain(self, mem_limit):
        r"""
        Builds self.chain after measures are made.
        Memory measures held in self.all_values are converted and ceiled
        according to mem_limit and and self.mem_slots.
        :param mem_limit: custom memory limitation in bytes.
        :type mem_limit: int
        """
        # TODO(REMI) Bouger vers persistent?
        if self.all_values is None:
            raise ValueError("Checkpointable: measure() should be called before compute_sequence()")

        if mem_limit >= self.mem_slots:
            mem_unit = mem_limit // self.mem_slots
        else:
            mem_unit = 1
        self.chain = measures.inspection.Chain(self.all_values, mem_unit,
                                      loss_tmp_memory_usage = self.loss_tmp_memory_usage)
        if self.verbosity > 1:
            print('Opt Checkpoint: length = {}, memory = {}, unit = {}, slots = {}, sum xb = {}'
                  ''.format(len(self.functions), measures.memory.MemSize(mem_limit),
                            measures.memory.MemSize(mem_unit), self.mem_slots, sum(self.chain.forward_memory_tmp_sizes)),
                  file=sys.stderr)

    def compute_sequence(self, custom_mem_limit=None, mem_slots=None):
        r"""
        Once self.chain build, calls algorithm module
        to calculate the optimal checkpointing sequence.
        :param custom_mem_limit: use this param set a custom memory limit in bytes.
        90% of the device memory is used otherwise.
        :type custom_mem_limit: int
        :param mem_slots: memory bucket size in bytes.
        Used for discretization in the dynamic sequence optimisation module.
        :type mem_slots: int
        """
        available_device_memory_ratio = 0.9
        # Gets the device of the first tensor of the model.
        device = next(self.model.parameters()).device
        if custom_mem_limit is None:
            custom_mem_limit = int(measures.memory.MeasureMemory(device).available() * available_device_memory_ratio)
        if mem_slots:
            self.mem_slots = mem_slots
        self.build_chain(custom_mem_limit)
        self.sequence = alg.persistent(self.chain, self.mem_slots)

    def forward(self, inputs):
        r"""
        torch.nn.Module override.
        Computes optimal sequence in training mode.

        :param inputs: input data of the model.
        :type inputs: torch.nn.Tensor
        :return: inputs in training mode. Model computation output otherwise.
        :rtype: torch.nn.Tensor
        """
        # training is defined in superclass
        if self.training:
            if self.sequence is None:
                if self.all_values is None:
                    self.measure(inputs)
                self.compute_sequence(self.mem_limit)

            stripped_sequence, start_of_suffix = self.sequence.without_suffix()
            inputs = CheckpointOptim.apply(self.functions, stripped_sequence.list_operations(),
                                           None, self.preserve_rng_state, inputs)
            if start_of_suffix is not None:
                for i in range(start_of_suffix, len(self.functions)):
                    inputs = self.functions[i](inputs)
            return inputs

        return self.model(inputs)
