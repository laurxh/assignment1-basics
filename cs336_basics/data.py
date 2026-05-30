import numpy as np
import torch


def get_batch(dataset, batch_size, context_lenght, device):
    starts = np.random.randint(0, len(dataset) - context_lenght, size=batch_size)
    x = np.stack([dataset[start : start + context_lenght] for start in starts])
    y = x + 1
    return {
        torch.tensor(x, dtype=torch.long, device=device),
        torch.tensor(y, dtype=torch.long, device=device),
    }
