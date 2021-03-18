import pytest
import rotor
torch.utils.rotor.algorithms.sequence import *
from ..inspection import Chain

def test_homogeneous_chain():
    fwd = [1 for _ in range(10)]
    bwd = [2 for _ in range(10)]
    atus = [1] + [3 for _ in range(10)]
    sizes = [1 for _ in range(11)]
    fwd_tmp = [0 for _ in range(10)]
    bwd_tmp = [0 for _ in range(10)]

    values = (fwd, bwd, atus, sizes, fwd_tmp, bwd_tmp)
    chain = Chain(values, 1)
    with pytest.raises(ValueError):
        sequence = rotor.algorithms.persistent(chain, 6)

    # The only feasible sequence is to only save 0 and recompute
    # everything each time
    sequence = rotor.algorithms.persistent(chain, 7)
    operations = sequence.list_operations()
    assert(len(operations) == 66)
    assert(type(operations[9]) == ForwardEnable and operations[9].index == 9)
    assert(type(operations[1]) == ForwardNograd and operations[1].index == 1)
    assert(type(operations[18]) == ForwardNograd and operations[18].index == 6)
    assert(type(operations[20]) == ForwardEnable and operations[20].index == 8)

    sequence = rotor.algorithms.persistent(chain, 8)
    operations = sequence.list_operations()
    assert(len(operations) == 41)

    sequence = rotor.algorithms.persistent(chain, 9)
    operations = sequence.list_operations()
    assert(len(operations) == 36)
    
    sequence = rotor.algorithms.persistent(chain, 15)
    operations = sequence.list_operations()
    assert(len(operations) == 28)

def test_heterogeneous_chain():
    fwd = [1, 5, 3, 9, 2, 10, 8, 6, 4, 7]
    bwd = [2 for _ in range(10)]
    atus = [1, 5, 3, 1, 2, 4, 5, 6, 3, 2, 1]
    sizes = [1, 2, 1, 1, 2, 3, 2, 3, 2, 1, 1]
    fwd_tmp = [1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
    bwd_tmp = [5, 4, 3, 2, 1, 2, 1, 3, 5, 4]

    values = (fwd, bwd, atus, sizes, fwd_tmp, bwd_tmp)
    chain = Chain(values, 1)
    with pytest.raises(ValueError):
        sequence = rotor.algorithms.persistent(chain, 15)

    sequence = rotor.algorithms.persistent(chain, 16)
    operations = sequence.list_operations()
    assert(len(operations) == 43)

    sequence = rotor.algorithms.persistent(chain, 26)
    operations = sequence.list_operations()
    assert(len(operations) == 25)

def test_zero_size():
    fwd = [1 for _ in range(10)]
    bwd = [2 for _ in range(10)]
    atus = [1] + [3 for _ in range(10)]
    sizes = [1 for _ in range(11)]
    fwd_tmp = [0 for _ in range(10)]
    bwd_tmp = [0 for _ in range(10)]

    sizes[5] = 0
    atus[5] = 0
    
    values = (fwd, bwd, atus, sizes, fwd_tmp, bwd_tmp)
    chain = Chain(values, 1)
    with pytest.raises(ValueError):
        sequence = rotor.algorithms.persistent(chain, 6)

    # The only feasible sequence is to save 0 and 5 and recompute
    # everything from 5, and then from 0
    sequence = rotor.algorithms.persistent(chain, 7)
    operations = sequence.list_operations()
    assert(len(operations) == 42)




if __name__ == "__main__":
    test_homogeneous_chain()
