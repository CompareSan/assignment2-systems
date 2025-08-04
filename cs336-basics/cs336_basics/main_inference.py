from cs336_basics.utils.generation import generate
from cs336_basics.tokenizers.bpe_tokenizer import BPETokenizer
from cs336_basics.layers.transformer_llm import Transformer
import torch



def main():
    vocab_size = 10_000
    context_len = 256
    num_layers = 4
    d_model = 512
    num_heads = 16
    d_ff = int(8/3 * d_model)
    theta = 10000.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Transformer(vocab_size, context_len, num_layers, d_model, num_heads, d_ff, theta)
    model.to(device)
    model = torch.compile(model)
    checkpoint = torch.load('./checkpoints/checkpoint_7000.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    tokenizer = BPETokenizer.from_files("./tokenizers/vocab.json", "./tokenizers/merges.txt", special_token=["<|endoftext|>"])

    prompt = "Once upon a time"
    max_tokens = 50

    generated_text = generate(model, tokenizer, prompt, max_tokens)
    print(generated_text)


if __name__ == "__main__":
    main()