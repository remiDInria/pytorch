import os
from torch.utils.cpp_extension import load
from torch.utils.rotor.algorithms import sequence as sq


is_os_cxx_preset = False
if 'CXX' in os.environ:
    is_os_cxx_preset = True
    original_default_compiler = os.environ['CXX']

# use default c compiler
os.environ['CXX'] = "cc"

module_path = os.path.dirname(__file__)
build_path = os.path.dirname(module_path)
dp = load(name="dynamic_programs", sources=[os.path.join(module_path, "dynamic_programs.c")],
          verbose=True,
          build_directory=build_path,
          extra_cflags=['-g3'])

if is_os_cxx_preset:
    os.environ['CXX'] = original_default_compiler
else:
    del os.environ['CXX']



# Builds the optimal sequence, recursive helper function
def persistent_rec(chain, lmin, lmax, cmem, opt_table):
    """ chain : the class describing the AC graph
        lmin : index of the first forward to execute
        lmax : upper bound index of the last forward to execute (not included)
        cmem : number of available memory slots
        Return the optimal sequence of makespan Opt_hete[cmem][lmin][lmax-lmin]"""
    opt, what = opt_table
    sequence = sq.Sequence(sq.Function("Persistent", lmax - lmin, cmem))

    if opt[cmem][lmin][lmax] == float("inf"):
        raise ValueError(
            "Can not process this chain from index {lmin} to {lmax} with memory {cmem}".format(lmin=lmin, lmax=lmax,
                                                                                               cmem=cmem))
    if lmin == lmax:
        if lmin == chain.length:
            sequence.insert(sq.Loss())
        else:
            sequence.insert(sq.ForwardEnable(lmin))
            sequence.insert(sq.Backward(lmin))
        return sequence

    if what[cmem][lmin][lmax][0]:
        sequence.insert(sq.ForwardEnable(lmin))
        sequence.insert_sequence(persistent_rec(chain, lmin + 1, lmax, cmem - chain.activation_total_usages[lmin + 1]
                                                , opt_table))
        sequence.insert(sq.Backward(lmin))
    else:
        j = what[cmem][lmin][lmax][1]
        sequence.insert(sq.ForwardCheck(lmin))
        for k in range(lmin + 1, j):
            sequence.insert(sq.ForwardNograd(k))
        sequence.insert_sequence(persistent_rec(chain, j, lmax, cmem - chain.activation_sizes[j], opt_table))
        sequence.insert_sequence(persistent_rec(chain, lmin, j - 1, cmem, opt_table))

    return sequence


# Computes the optimal sequence for the given parameters
def persistent(chain, memory_limit):
    if memory_limit <= chain.activation_sizes[0]:
        raise ValueError("Can not process a sequence if available memory {cmem} "
                         "is not more than size of input {input_size}"
                         "".format(cmem=memory_limit, input_size=chain.activation_sizes[0]))

    opt_table = dp.persistent_compute_table(chain, memory_limit)
    return persistent_rec(chain, 0, chain.length, memory_limit - chain.activation_sizes[0], opt_table)
