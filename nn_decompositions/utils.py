import torch


def get_free_gpu() -> str:
    """Return the CUDA device with the most free memory, or 'cpu' if no GPU."""
    if not torch.cuda.is_available():
        return "cpu"
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    return f"cuda:{free.index(max(free))}"
