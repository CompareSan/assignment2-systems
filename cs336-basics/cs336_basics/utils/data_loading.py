import numpy as np
import torch


def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str = 'cpu'):

    max_start = len(x) - context_length
    
    starts = np.random.randint(0, max_start, size=batch_size)
    
    inputs = np.stack([x[i : i + context_length] for i in starts])
    targets = np.stack([x[i + 1 : i + 1 + context_length] for i in starts])
    
    inputs_t = torch.tensor(inputs, dtype=torch.long, device=device)
    targets_t = torch.tensor(targets, dtype=torch.long, device=device)
    return inputs_t, targets_t

def get_shuffled_batches(x: np.ndarray, batch_size: int, context_length: int, device: str = 'cpu'):
    """
    Generates shuffled batches for one epoch.
    """
    num_contexts = len(x) - context_length
    all_indices = np.arange(num_contexts)
    np.random.shuffle(all_indices)
    
    for i in range(0, num_contexts, batch_size):
        batch_indices = all_indices[i:i+batch_size]
        
        # Drop last batch if it's smaller than batch_size
        if len(batch_indices) < batch_size:
            continue
            
        inputs = np.stack([x[j : j + context_length] for j in batch_indices])
        targets = np.stack([x[j + 1 : j + 1 + context_length] for j in batch_indices])
        
        inputs_t = torch.tensor(inputs, dtype=torch.long, device=device)
        targets_t = torch.tensor(targets, dtype=torch.long, device=device)
        
        yield inputs_t, targets_t