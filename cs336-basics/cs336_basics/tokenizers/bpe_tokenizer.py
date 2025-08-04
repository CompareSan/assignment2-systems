import json
from typing import Iterable, Iterator
from cs336_basics.tokenizers.pretokenizer import Pretokenizer


class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None) -> None:
        self.vocab = vocab
        self.swapped_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens
        self._pretokenizer = Pretokenizer(special_tokens)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_token: list[str] | None = None) -> 'BPETokenizer':
        with open(vocab_filepath, "r") as vocab_file:
            vocab = json.load(vocab_file)
            vocab = {int(k): bytes.fromhex(v) for k, v in vocab.items()}
        with open(merges_filepath, "r") as merges_file:
            merges = [tuple(map(bytes.fromhex, line.strip().split())) for line in merges_file]
        return BPETokenizer(vocab, merges, special_token)

    
    def encode(self, text: str) -> list[int]:
        pretokens = self._pretokenizer._pretoken_text(text)
        encoding: list[int] = []

        for pretoken in pretokens:
            if self.special_tokens and pretoken in self.special_tokens:
                encoding.append(self.swapped_vocab[pretoken.encode('utf-8')])
                continue

            symbols = [bytes([b]) for b in pretoken.encode("utf-8")]

            for merge in self.merges:
                i = 0
                while i < len(symbols) - 1:
                    if symbols[i] == merge[0] and symbols[i + 1] == merge[1]:
                        symbols[i:i+2] = [symbols[i] + symbols[i+1]]
                    else:
                        i += 1

            for s in symbols:
                encoding.append(self.swapped_vocab[s])

        return encoding

    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, tokens: list[int]) -> str:
        byte_sequence = b''.join(self.vocab[token] for token in tokens)
        return byte_sequence.decode('utf-8')



if __name__ == "__main__":
    import numpy as np
    import tqdm
    tokenizer = BPETokenizer.from_files("vocab.json", "merges.txt", special_token=["<|endoftext|>"])
    valid_ids = []
    with open("../data/TinyStoriesV2-GPT4-valid.txt") as f:
        for _id in tqdm.tqdm(tokenizer.encode_iterable(f), desc="Encoding"):
            valid_ids.append(_id)
    valid_ids_np = np.array(valid_ids, dtype=np.uint16)
    np.save("valid_ids.npy", valid_ids_np)