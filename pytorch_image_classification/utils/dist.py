import torch.distributed as dist


def get_rank() -> int:
    # Return rank 0 outside distributed mode so callers can use this helper without
    # branching on whether torch.distributed has been initialized.
    if not (dist.is_available() and dist.is_initialized()):
        return 0
    else:
        return dist.get_rank()
