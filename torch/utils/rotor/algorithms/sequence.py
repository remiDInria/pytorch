
# Representation of the optimal sequences computed by
# the optimal rotor algorithm.

# All possible types of operations in a sequence
class _Operation:
    def __init__(self, index=-1, name="O"):
        self.index = index
        self.name = name

    def __repr__(self):
        return "{n}_{i}".format(n=self.name, i=self.index)


class Forward(_Operation):
    def __init__(self, index, name="F"):
        super().__init__(index=index, name=name)


class ForwardEnable(Forward):
    def __init__(self, index):
        super().__init__(index=index, name="Fe")


class ForwardNograd(Forward):
    def __init__(self, index):
        super().__init__(index=index, name="Fn")


class ForwardCheck(Forward):
    def __init__(self, index):
        super().__init__(index=index, name="CF")


class Backward(_Operation):
    def __init__(self, index):
        super().__init__(index=index, name="B")


class Loss(_Operation):
    def __init__(self):
        super().__init__(name="L")

    def __repr__(self):
        return "L"

# The Function class represents how a particular sequence was obtained
# (which algorithm, which parameters)
class Function:
    def __init__(self, name, *args):
        self.name = name
        self.args = args
        self.str_args = ','.join(str(v) for v in self.args)

    def __repr__(self):
        return "{n}({args})".format(n=self.name, args=self.str_args)

# The main Sequence class, which contains all Operations, and retains
# in its structure the suite of recursive calls made to obtain the sequence
class Sequence:
    def __init__(self, function):
        self.sequence = []  # List of Operation and Sequence
        self.function = function  # Description the function (name and parameters)

    def __repr__(self):
        return repr(self.list_operations())

    def list_operations(self):
        operation_list = []
        for x in self.sequence:
            if isinstance(x, _Operation):
                operation_list.append(x)
            else:
                assert isinstance(x, Sequence)
                operation_list.extend(x.list_operations())
        return operation_list

    def insert(self, operation):
        self.sequence.append(operation)

    def insert_sequence(self, sequence):
        self.sequence.append(sequence)

    # If the Forward phase of this Sequence finishes with a
    # consecutive segment of Fe operations, returns a tuple of a new
    # sequence with that segment stripped away and the index of the
    # first Fe operation in that segment.
    def without_suffix(self):
        ops = self.list_operations()
        end_of_first_phase = [i for i in range(len(ops)) if type(ops[i]) is Loss][0]
        try:
            # Index of the last operation which is not a ForwardEnable
            last_index = max(i for i in range(end_of_first_phase) if not type(ops[i]) is ForwardEnable)
        except ValueError:
            last_index = -1
        if last_index == end_of_first_phase - 1:
            return (self, None)
        # We can safely assume that the segment finished with a
        # forward operation on the last layer
        chain_length = ops[end_of_first_phase - 1].index
        start_of_fwd_enable_chain = ops[last_index + 1].index
        result = Sequence(Function("Strip", self.function.name, *self.function.args, start_of_fwd_enable_chain))
        for i in range(last_index + 1):
            result.insert(ops[i])
        result.insert(Loss())
        for i in range(chain_length, start_of_fwd_enable_chain - 1, -1):
            # Check that the removed operations have the expected structure
            position = end_of_first_phase + 1 + (chain_length - i)
            assert type(ops[position]) is Backward
            assert ops[position].index == i
        for i in range(end_of_first_phase + 1 + 1 + chain_length - start_of_fwd_enable_chain, len(ops)):
            result.insert(ops[i])
        return (result, start_of_fwd_enable_chain)
