import regex as re
from collections import defaultdict
import time

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(input_path, vocab_size, special_tokens) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    if special_tokens:
        special_pattern = "|".join(re.escape(tok) for tok in special_tokens)
        chunks = re.split(special_pattern, text)
    else:
        chunks = [text]
    merges = []
    vocab = {}
    word_counts = defaultdict(int)
    for i in range(256):
        vocab[len(vocab)] = bytes([i])
    pair_counts = defaultdict(int)
    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            token = match.group()
            token_bytes = token.encode("utf-8")
            word = tuple(bytes([b]) for b in token_bytes)
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i + 1])] += 1
            word_counts[word] += 1
    # total_count_time = 0
    # total_max_time = 0
    # total_update_time = 0

    while len(vocab) < vocab_size - len(special_tokens):
        t1 = time.time()
        max_pair = None
        for pair, cnt in pair_counts.items():
            if max_pair == None:
                max_pair = (cnt, pair)
            else:
                max_pair = max(max_pair, (cnt, pair))
        merges.append(max_pair[1])
        vocab[len(vocab)] = max_pair[1][0] + max_pair[1][1]
        new_word_counts = defaultdict(int)
        # t3 = time.time()

        for word, cnt in word_counts.items():
            new_word = []
            i = 0
            last_merge = -1
            while i < len(word):
                if i + 1 < len(word) and (word[i], word[i + 1]) == max_pair[1]:
                    new_word.append(word[i] + word[i + 1])
                    if last_merge != -1:
                        pair_counts[(word[i - 1], word[i])] -= cnt
                    pair_counts[(word[i], word[i + 1])] -= cnt
                    if last_merge == 0:
                        pair_counts[word[i - 1], new_word[-1]] += cnt
                    elif last_merge == 1:
                        pair_counts[new_word[-2], new_word[-1]] += cnt
                    i += 1
                    last_merge = 1
                else:
                    new_word.append(word[i])
                    if last_merge == 1:
                        pair_counts[(word[i - 1], word[i])] -= cnt
                        pair_counts[(new_word[-2], new_word[-1])] += cnt
                    last_merge = 0
                i += 1
            new_word_counts[tuple(new_word)] = cnt
        pair_counts = defaultdict(int, {k: v for k, v in pair_counts.items() if v > 0})
        word_counts = new_word_counts
        # t4 = time.time()
        # total_max_time += t3 - t1
        # total_update_time += t4 - t3
    # print(f"count_time:{total_count_time}")
    # print(f"max_time:{total_max_time}")
    # print(f"update_time:{total_update_time}")
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    return vocab, merges


if __name__ == "__main__":
    print(bytes([2]))
    text = "Hello, I'm 123! 你好 world.\n"

    print("original text:")
    print(repr(text))
    print()

    print("pre-tokenized:")
    for match in re.finditer(PAT, text):
        token = match.group()
        token_bytes = token.encode("utf-8")
        byte_tokens = tuple(bytes([b]) for b in token_bytes)

        print(f"{token!r} -> {token_bytes!r} -> {byte_tokens}")
