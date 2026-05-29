import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path, vocab_size, special_tokens
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    if special_tokens:
        special_pattern = "|".join(re.escape(tok) for tok in special_tokens)
        chunks = re.split(special_pattern, text)
    else:
        chunks = [text]

    merges = []
    vocab = {}
    word_counts = {}
    for token in special_tokens:
        vocab[len(vocab)] = token
    for i in range(256):
        vocab[len(vocab)] = bytes([i])
    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            token = match.group()
            token_bytes = token.encode("utf-8")
            word = tuple(bytes([b]) for b in token_bytes)
            word_counts[word] += 1
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
