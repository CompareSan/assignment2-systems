import torch 
from cs336_basics.optimizers.adam import AdamW
from cs336_basics.layers.transformer_llm import Transformer
from cs336_basics.trainer import Trainer
from cs336_basics.utils.save_model import load_checkpoint
import numpy as np
import os




def main():   
    vocab_size = 10_000
    context_len = 256
    num_layers = 4
    d_model = 512
    num_heads = 16
    d_ff = int(8/3 * d_model)
    theta = 10000.0
    batch_size = 16
    n_steps = 5000  # number of token processed n_steps * batch_size * context_len (approx 250M), total FLOPS approx 250M * n_parameters (22M) * 2 = 1.12 e16 FLOPS
    lr = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # if checkpoint folder is not empty, load last checkpoint:
    model = Transformer(vocab_size, context_len, num_layers, d_model, num_heads, d_ff, theta)
    model.to(device)
    model = torch.compile(model)
    optimizer = AdamW(model.parameters(), lr=lr)

    try:
        last_checkpoint = sorted(os.listdir('./checkpoints'))[-1]
        iteration = load_checkpoint(f'./checkpoints/{last_checkpoint}', model=model, optimizer=optimizer, device=device)
    except (FileNotFoundError, IndexError):
        iteration = 0
    
    trainer = Trainer(model, optimizer, device=device, iteration=iteration,
                      )


    dataset = np.load('./data/valid_ids.npy')  # Load your dataset here

    train_losses = trainer.train(dataset, batch_size, context_len, n_steps, lr_max=lr, lr_min=1e-5, t_warmup=1000, t_post=4000)

    print("Training complete. Final training loss:", train_losses[-1])


if __name__ == "__main__":
    main()