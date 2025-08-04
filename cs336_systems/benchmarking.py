from cs336_basics.layers.transformer_llm import Transformer
from cs336_basics.optimizers.adam import AdamW
from cs336_basics.losses.cross_entropy_loss import cross_entropy_loss
from cs336_basics.trainer import Trainer
import torch
import timeit
import argparse



def benchmark_model(model, data, batch_size, n_steps, warmup_steps, forward_only=True, device='cuda'):
    """
    Benchmark the model with forward and optionally backward passes.
    
    Args:
        model: The transformer model to benchmark
        data: Input data tensor
        batch_size: Batch size for training
        n_steps: Number of steps to time (after warmup)
        warmup_steps: Number of warmup steps before timing
        forward_only: If True, only run forward pass. If False, run forward+backward
        device: Device to run on
    
    Returns:
        Average time per step in seconds
    """
    model.train() if not forward_only else model.eval()
    
    print(f"Running {'forward-only' if forward_only else 'forward+backward'} benchmark...")
    print(f"Warmup steps: {warmup_steps}, Timing steps: {n_steps}")
    
    optimizer = AdamW(model.parameters())

    # Warmup phase
    print("Starting warmup...")
    for i in range(warmup_steps):
        batch_start = (i * batch_size) % (data.shape[0] - batch_size)
        batch_data = data[batch_start:batch_start + batch_size].to(device)
        
        if forward_only:
            with torch.no_grad():
                logits = model(batch_data)
        else:

            optimizer.zero_grad()
            # Forward pass
            logits = model(batch_data)
            # Simple loss for backward pass (cross-entropy with random targets)
            targets = torch.randint(0, logits.size(-1), (batch_size, logits.size(1)), device=device)
            loss = cross_entropy_loss(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
            # Backward pass
            loss.backward()
            optimizer.step()
            
        if device == 'cuda':
            torch.cuda.synchronize()
    
    print("Warmup complete. Starting timed benchmark...")
    
    # Timed benchmark phase
    start_time = timeit.default_timer()
    
    for i in range(n_steps):
        batch_start = ((warmup_steps + i) * batch_size) % (data.shape[0] - batch_size)
        batch_data = data[batch_start:batch_start + batch_size].to(device)
        
        if forward_only:
            with torch.no_grad():
                logits = model(batch_data)
        else:
            # Forward pass
            logits = model(batch_data)
            # Simple loss for backward pass
            targets = torch.randint(0, logits.size(-1), (batch_size, logits.size(1)), device=device)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
            # Backward pass
            loss.backward()
            model.zero_grad()
        
        if device == 'cuda':
            torch.cuda.synchronize()
    
    end_time = timeit.default_timer()
    
    total_time = end_time - start_time
    avg_time_per_step = total_time / n_steps
    
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average time per step: {avg_time_per_step:.6f} seconds")
    print(f"Steps per second: {1.0 / avg_time_per_step:.2f}")
    
    return avg_time_per_step


def main():   
    parser = argparse.ArgumentParser(description='Benchmark Transformer model')
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--d_model', type=int, default=768, help='Model dimension')
    parser.add_argument('--d_ff', type=int, default=3072, help='Feed forward dimension')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--context_len', type=int, default=256, help='Context length')
    parser.add_argument('--theta', type=float, default=10000.0, help='RoPE theta parameter')
    parser.add_argument('--n_steps', type=int, default=10, help='Number of timing steps')
    parser.add_argument('--warmup_steps', type=int, default=5, help='Number of warmup steps')
    parser.add_argument('--forward_only', action='store_true', help='Only run forward pass')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Initialize model
    print("Initializing model...")
    model = Transformer(
        vocab_size=args.vocab_size,
        context_len=args.context_len,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta
    )
    model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.2f} MB (assuming float32)")
    
    # Generate random dataset (make it large enough for multiple batches)
    dataset_size = max(1000, (args.warmup_steps + args.n_steps) * args.batch_size + args.batch_size)
    print(f"Generating random dataset of size {dataset_size}...")
    dataset = torch.randint(0, args.vocab_size, (dataset_size, args.context_len), dtype=torch.long)
    
    # Run benchmark
    avg_time = benchmark_model(
        model=model,
        data=dataset,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        warmup_steps=args.warmup_steps,
        forward_only=args.forward_only,
        device=device
    )
    
    print(f"\nBenchmark complete!")
    print(f"Average time per step: {avg_time:.6f} seconds")

if __name__ == "__main__":
    main()




