import os
import psutil
import torch
import subprocess
from torch.utils.rotor import inspection
from torch.utils.rotor import timing

__all__ = ["MemSize", "MeasureMemory", "DisplayMemory"]

def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


class MemSize:
    def __init__(self, v):
        self.v = v

    def __add__(self, other):
        return self.__class__(self.v + other.v)

    def __sub__(self, other):
        return self.__class__(self.v - other.v)

    @classmethod
    def fromStr(cls, str):
        suffixes = {'k': 1024, 'M': 1024 * 1024, 'G': 1024 * 1024 * 1024}
        if str[-1] in suffixes:
            val = int(float(str[:-1]) * suffixes[str[-1]])
        else:
            val = int(str)
        return MemSize(val)

    def __str__(self):
        return sizeof_fmt(self.v)

    def __format__(self, fmt_spec):
        return sizeof_fmt(self.v).__format__(fmt_spec)

    def __repr__(self):
        return str(self.v)

    def __int__(self):
        return self.v


class MeasureMemory:
    def __init__(self, device):
        self.device = device
        self.cuda = self.device.type == 'cuda'
        if self.cuda:
            self.max_memory = torch.cuda.max_memory_allocated(self.device)
        else:
            self.process = psutil.Process(os.getpid())
            self.max_memory = 0
        self.last_memory = self.current_value()
        self.start_memory = self.last_memory

    def current_value(self):
        """
        Returns the resident set size of the process on cpu.
        or the memory allocated on the current cuda device.
        Updates self.max_memory if cpu is used.
        """
        if self.cuda:
            result = torch.cuda.memory_allocated(self.device)
        else:
            result = int(self.process.memory_info().rss)
            self.max_memory = max(self.max_memory, result)
        return result

    def maximum_value(self):
        if self.cuda:
            return MemSize(torch.cuda.max_memory_allocated(self.device))
        else:
            return MemSize(self.max_memory)

    # Requires Pytorch >= 1.1.0
    def available(self, gpu_device_index=None):
        if not self.cuda:
            return psutil.virtual_memory().available

        csv_gpu_free_memories_mb = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"])
        gpu_free_memories_mb = [int(x) for x in csv_gpu_free_memories_mb.strip().split(b"\n")]
        if gpu_device_index is None:
            gpu_device_index = self.device.index
        if gpu_device_index is None:
            gpu_device_index = torch.cuda.current_device()
        # TODO:REMI precisions
        # trous à réallouer (mémoire possiblement dispo)
        return gpu_free_memories_mb[gpu_device_index] * 1024 * 1024 + torch.cuda.memory_cached(self.device) - torch.cuda.memory_allocated(self.device)

    ## Requires Pytorch >= 1.1.0
    def reset_max_memory(self):
        if self.cuda:
            torch.cuda.reset_max_memory_allocated(self.device)
        else:
            self.current_value()

    def current_mem_usage(self):
        return MemSize(self.current_value())

    def diff_from_last_and_reset_last_memory(self):
        """
        Calculates memory usage since last measure.
        Sets self.last_memory to current value.
        Updates self.max_memory if cpu is used.
        :return: diff between current mem usage and last value recorded.
        """
        current = self.current_value()
        result = current - self.last_memory
        self.last_memory = current
        return MemSize(result)

    def diff_from_start(self):
        current = self.current_value()
        return MemSize(current - self.start_memory)

    def current_cached(self):
        """
        TODO : only used for printout.
        :return: 0 if cpu, cuda cached memory else.
        """
        if not self.cuda:
            return 0
        else:
            return MemSize(torch.cuda.memory_cached(self.device))

    def measure(self, func, *args):
        """
        Measures the memory usage of func application.
        :param func: python function to be measured.
        :param args: func arguments.
        :return: function results, memory and maximum memory usages in bytes
        of the function application on args.
        """
        self.diff_from_last_and_reset_last_memory()
        self.reset_max_memory()
        max_memory_before = self.maximum_value()
        func_result = func(*args)
        func_memory_usage = self.diff_from_last_and_reset_last_memory()
        func_max_memory_usage = self.maximum_value() - max_memory_before

        return func_result, func_memory_usage, func_max_memory_usage


class DisplayMemory:
    def __init__(self, device, maxLabelSize=45):
        self.device = device
        self.memUsage = MeasureMemory(device)
        self.setMaxLabelSize(maxLabelSize)
        self.progress = None

    def setMaxLabelSize(self, size):
        self.maxLabelSize = size
        self.formatStringTime = \
            "{:<%d} {:>7.2f} TotalMem: {:>12} max reached: {:>12} wrt to last: {:>12} cached: {:>12}" \
            % self.maxLabelSize
        self.formatStringNoTime = \
            "{:<%d}         TotalMem: {:>12} max reached: {:>12} wrt to last: {:>12} cached: {:>12}" \
            % self.maxLabelSize

    def printCurrentState(self, *args, **kwargs):
        if self.progress:
            self.progress.startFwd(None)
        self._printCurrentState(*args, **kwargs)

    def _printCurrentState(self, label, time=None):
        current = self.memUsage.current_mem_usage()
        maxUsed = self.maximumValue()
        fromLast = self.memUsage.diff_from_last_and_reset_last_memory()
        cached = self.memUsage.current_cached()
        if time:
            print(self.formatStringTime.format(label, time, current, maxUsed, fromLast, cached))
        else:
            print(self.formatStringNoTime.format(label, current, maxUsed, fromLast, cached))

    def maximumValue(self):
        return self.memUsage.maximum_value()

    def inspectModule(self, module):
        self.progress = timing.ProgressTimer(timing.make_timer(self.device), self._printCurrentState)
        maxLength = 0
        for (name, m) in inspection.extract_children_from_sequential(module):
            maxLength = max(maxLength, len(name))
            m.register_forward_hook(lambda x, y, z, n = name: self.progress.endFwd(n))
            m.register_forward_pre_hook(lambda x, y, n = name: self.progress.startFwd(n))
            m.register_backward_hook(lambda x, y, z, n = name: self.progress.endBwd(n))
        self.setMaxLabelSize(maxLength + self.progress.additionalLength)
        self.progress.startFwd(None)

        # ## For more inspection if desired
        # for (name, p) in module.named_parameters(): 
        #     p.register_hook(lambda g, m = name: self._printCurrentState("Param " + m))
