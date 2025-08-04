import torch
import torch.nn as nn
from cs336_basics.tokenizers.bpe_tokenizer import BPETokenizer
from cs336_basics.layers.softmax import softmax

def generate(model: nn.Module, tokenizer: BPETokenizer, prompt: str, max_tokens: int, top_p: float = 0.95, t: float = 1.0):
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt)
    generated_ids = prompt_ids.copy()

    for _ in range(max_tokens):
        input_ids = torch.tensor(generated_ids, device=device, dtype=torch.long).unsqueeze(0)
        logits = model(input_ids)[:, -1, :]
        probas = softmax(logits, dim=-1, t=t).squeeze(0)
    

        sorted_probas, sorted_indices = torch.sort(probas, descending=True)
        cumulative_probas = torch.cumsum(sorted_probas, dim=0)
        cutoff = cumulative_probas <= top_p
        filtered_probas = sorted_probas[cutoff]
        filtered_probas = filtered_probas / filtered_probas.sum()

        if filtered_probas.numel() == 0:
            new_token = sorted_indices[0].item()
        else:
            chosen_idx = torch.multinomial(filtered_probas, replacement=True, num_samples=1)
            new_token = sorted_indices[chosen_idx].item()


        if tokenizer.decode([new_token]) in tokenizer.special_tokens:
            break

        generated_ids.append(new_token)

    return tokenizer.decode(generated_ids[len(prompt_ids):])
       




