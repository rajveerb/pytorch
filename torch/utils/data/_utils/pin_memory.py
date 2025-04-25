r""""Contains definitions of the methods used by the _BaseDataLoaderIter to put
fetched tensors into pinned memory.

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import collections
import queue
import heapq

import torch
from . import MP_STATUS_CHECK_INTERVAL
from torch._utils import ExceptionWrapper


def _pin_memory_loop(in_queue, out_queue, device_id, done_event, device, log_file=None, pid=None):
    # This setting is thread local, and prevents the copy in pin_memory from
    # consuming all CPU cores.
    torch.set_num_threads(1)

    if device == "cuda":
        torch.cuda.set_device(device_id)
    elif device == "xpu":
        torch.xpu.set_device(device_id)  # type: ignore[attr-defined]

    rcv_index = 0
    pid = pid if pid is not None else 0
    if log_file:
        import psutil
        pid = psutil.Process().pid

    priority_heap = []

    def do_one_step():
        nonlocal rcv_index
        try:
            r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            idx, data = r
            heapq.heappush(priority_heap, (idx, data))
        except queue.Empty:
            pass  # No new batch in the input queue

        # Try to process the batch with the lowest ID (highest priority)
        if priority_heap:
            idx, data = heapq.heappop(priority_heap)
            if not done_event.is_set() and not isinstance(data, ExceptionWrapper):
                try:
                    if log_file:
                        import time
                        start = time.time_ns()
                    data = pin_memory(data, device)
                    if log_file:
                        import time
                        end = time.time_ns()
                        import psutil
                        with open(log_file+f"_main_pid_{pid}", "a") as f:
                            f.write(f'SBatchPinMemory_{idx},{start},{end-start}\n')
                except Exception:
                    data = ExceptionWrapper(
                        where="in pin memory thread for device {}".format(device_id))
                r = (idx, data)
            else:
                r = (idx, data)
            while not done_event.is_set():
                try:
                    out_queue.put(r, timeout=MP_STATUS_CHECK_INTERVAL)
                    break
                except queue.Full:
                    continue

    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.
    while not done_event.is_set():
        # Make sure that we don't preserve any object from one iteration
        # to the next
        do_one_step()

def pin_memory(data, device=None):
    if isinstance(data, torch.Tensor):
        return data.pin_memory(device)
    elif isinstance(data, (str, bytes)):
        return data
    elif isinstance(data, collections.abc.Mapping):
        try:
            return type(data)({k: pin_memory(sample, device) for k, sample in data.items()})  # type: ignore[call-arg]
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {k: pin_memory(sample, device) for k, sample in data.items()}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return type(data)(*(pin_memory(sample, device) for sample in data))
    elif isinstance(data, tuple):
        return [pin_memory(sample, device) for sample in data]  # Backwards compatibility.
    elif isinstance(data, collections.abc.Sequence):
        try:
            return type(data)([pin_memory(sample, device) for sample in data])  # type: ignore[call-arg]
        except TypeError:
            # The sequence type may not support `__init__(iterable)` (e.g., `range`).
            return [pin_memory(sample, device) for sample in data]
    elif hasattr(data, "pin_memory"):
        return data.pin_memory()
    else:
        return data
