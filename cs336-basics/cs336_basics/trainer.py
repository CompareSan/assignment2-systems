import torch
import torch.nn as nn
import numpy as np
from cs336_basics.utils.data_loading import get_shuffled_batches
from cs336_basics.utils.save_model import save_checkpoint
from cs336_basics.losses.cross_entropy_loss import cross_entropy_loss
import tqdm
from cs336_basics.optimizers.lr_scheduler import cosine_annealing_scheduler
from cs336_basics.optimizers.gradient_clipping import gradient_clipping

class Trainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, device: str = 'cpu', iteration: int = 0, max_grad_norm: float = 1.0,
                 ) -> None:
    

        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.iteration = iteration
        self.max_grad_norm = max_grad_norm
    
    def train(self, dataset: np.ndarray, batch_size: int, context_len: int, n_steps: int,
              lr_max: float, lr_min: float, t_warmup: int, t_post: int
              ) -> list[float]:
        all_losses = []
        self.model.train()
        
        pbar = tqdm.tqdm(range(n_steps), desc="Training", unit="step")
        
        batch_generator = get_shuffled_batches(dataset, batch_size, context_len, self.device)

        for _ in pbar:
            new_lr = cosine_annealing_scheduler(self.iteration, lr_max, lr_min, t_warmup, t_post)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

            try:
                inputs, targets = next(batch_generator)
            except StopIteration:
                # Reset the generator if it's exhausted
                batch_generator = get_shuffled_batches(dataset, batch_size, context_len, self.device)
                inputs, targets = next(batch_generator)

            loss = self._train_step(inputs, targets)
            all_losses.append(loss)
            self.iteration += 1
            pbar.set_postfix(loss=f"{loss:.4f}")

            if self.iteration > 0 and self.iteration % 1000 == 0:
                self.save(f'./checkpoints/checkpoint_{self.iteration}.pt')

        self.save('./checkpoints/final_checkpoint.pt')
        return all_losses
        

    def _train_step(self, inputs: torch.Tensor, targets: torch.Tensor):
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = cross_entropy_loss(outputs, targets)
        loss.backward()
        gradient_clipping(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss.item()
    
    def eval(self, dataset: np.ndarray, batch_size: int, context_len: int) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            batch_generator = get_shuffled_batches(dataset, batch_size, context_len, self.device)
            for inputs, targets in batch_generator:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = cross_entropy_loss(outputs, targets)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches
    
    def save(self, out: str) -> None:
        save_checkpoint(self.model, self.optimizer, self.iteration, out)
