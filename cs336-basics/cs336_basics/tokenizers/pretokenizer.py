from typing import BinaryIO
from collections import Counter
import os
import regex as re
from multiprocessing import Pool, cpu_count

class Pretokenizer:
    def __init__(self, special_tokens: list[str]) -> None:
        self.special_tokens = special_tokens
        self.pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def _get_pretoken_counts(
        self,
        input_path: str | os.PathLike,
    ) -> Counter:
        with open(input_path, "rb") as f:
            boundaries = self._find_chunk_boundaries(f, 100, self.special_tokens[0].encode("utf-8"))

            pretoken_counts: Counter = Counter()
            # Escape special tokens for regex
            escaped_tokens = list(map(re.escape, self.special_tokens))
            split_pattern = "|".join(escaped_tokens)

            tasks = [
                (input_path, start, end, split_pattern, self.pat)
                for start, end in zip(boundaries[:-1], boundaries[1:])
            ]
            with Pool(cpu_count()) as pool:
                results = pool.map(self._process_chunk, tasks)
        
            for counter in results:
                pretoken_counts.update(counter)

            return self._convert_string_to_bytes(pretoken_counts)

    def _process_chunk(self, args):
        """Process a single chunk and return its token counts."""
        file_path, start, end, split_pattern, PAT = args
        counter = Counter()
        with open(file_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            parts = re.split(split_pattern, chunk)
            for part in parts:
                for match in re.finditer(PAT, part):
                    token = match.group(0)
                    counter[token] += 1
        return counter

    def _find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))
    
    def _convert_string_to_bytes(
        self,
        token_counts: Counter,
    ) -> Counter:
        """
        Converts string tokens to tuples of bytes for initial BPE vocabulary.
        """
        return Counter({tuple(token.encode("utf-8")): count for token, count in token_counts.items()})
    
    def _pretoken_text(self, text: str) -> list[str]:
        """
        Pre-tokenizes the input text using the defined pre-tokenization pattern.
        """
        if not self.special_tokens:
            return re.findall(self.pat, text)

        # Sort special tokens by length (descending) to prioritize longer matches
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        special_pattern = "|".join(map(re.escape, sorted_special_tokens))

        # Split the text by special tokens, keeping the delimiters
        parts = re.split(f"({special_pattern})", text)

        pre_tokens = []
        for part in parts:
            if part in self.special_tokens:
                pre_tokens.append(part)
            else:
                pre_tokens.extend(re.findall(self.pat, part))
        
        return pre_tokens