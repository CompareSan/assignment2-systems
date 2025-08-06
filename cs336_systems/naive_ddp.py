import os
import torch 
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import ReduceOp
import torch.multiprocessing as mp

from cs336_systems.simple_mlp import MLP




def load_data(rank: int, world_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = 512
    local_batch_size = batch_size // world_size
    num_dim = 1024
    torch.manual_seed(42 + rank)  # Ensure different data for each rank
    return torch.randn(local_batch_size, num_dim), torch.randn(local_batch_size, 1)


def setup(rank: int, world_size: int, backend: str):
    os.environ["MASTER_ADDR"] = "localhost" # Master run the process with rank 0
    os.environ["MASTER_PORT"] = "29500"
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

# this function gets run async by all process in parallel on a given batch
def simple_ddp(rank: int, world_size: int, model: nn.Module, n_steps: int, criterion: nn.Module, backend: str): # data is of batch_size // world_size along the 0 dim
    setup(rank, world_size, backend) # Add the process at rank to the process group
    
    if backend == "nccl":
        device: torch.device = "cuda"
    else:
        device = "cpu"

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001) # each rank has its own optimizer state

    for _ in range(n_steps): 
        
        inputs, targets = load_data(rank, world_size) 
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        y = model(inputs)
        
        loss = criterion(y, targets)

        loss.backward() # Compute gradients

        # Sync gradients for each parameter
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=ReduceOp.SUM, async_op=False)
                param.grad /= world_size  # Manually average

        optimizer.step()

        print(f"Rank {rank}, Step {_}, Loss: {loss.item()}, params = {[p.grad.norm().item() for p in model.parameters() if p.grad is not None]}")


if __name__ == "__main__":
    world_size = 4
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    # Example usage
    input_dim = 1024
    hidden_dim = 512
    output_dim = 1

    model = MLP(input_dim, hidden_dim, output_dim)

    criterion = nn.MSELoss()
    

    n_steps = 1
    # i want each rank to only load it's own data
    mp.spawn(simple_ddp, args=(world_size, model, n_steps, criterion, backend), nprocs=world_size, join=True)

    print("Training complete.")










