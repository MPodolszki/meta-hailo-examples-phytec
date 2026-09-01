"""Minimal Whisper token decoder.

Only the decode direction is needed (ids -> text), so this reimplements the
GPT-2 byte-level BPE detokenizer against the plain ``vocab.json`` instead of
pulling in `transformers` + `tokenizers`, neither of which is packaged for the
target and both of which would try to reach the network on first use.
"""

import json
import re

# First special token of the multilingual Whisper vocabulary (<|endoftext|>).
# Everything from here up is a control token: language, task, timestamps, ...
SPECIAL_TOKEN_START = 50257

EOT_TOKEN = 50257

# Prompt prefix Whisper expects before free generation:
# <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>
FORCED_DECODER_IDS = [50258, 50259, 50359, 50363]


def _bytes_to_unicode():
    """GPT-2's reversible byte <-> unicode mapping."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    mapped = list(printable)
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            mapped.append(256 + extra)
            extra += 1
    return dict(zip(printable, (chr(c) for c in mapped)))


class WhisperTokenizer:
    """Byte-level BPE detokenizer for the Whisper vocabulary."""

    def __init__(self, vocab_path):
        with open(str(vocab_path), "r", encoding="utf-8") as handle:
            vocab = json.load(handle)

        self.id_to_token = {int(index): token for token, index in vocab.items()}
        self.byte_decoder = {ch: byte for byte, ch in _bytes_to_unicode().items()}

    def decode(self, token_ids, skip_special_tokens=True):
        pieces = []
        for token_id in token_ids:
            token_id = int(token_id)
            if skip_special_tokens and token_id >= SPECIAL_TOKEN_START:
                continue
            token = self.id_to_token.get(token_id)
            if token is not None:
                pieces.append(token)

        text = "".join(pieces)
        raw = bytes(self.byte_decoder[ch] for ch in text if ch in self.byte_decoder)
        return raw.decode("utf-8", errors="replace")


def clean_transcription(text):
    """Drop repeated sentences, which greedy decoding without a beam produces."""
    text = text.strip()
    if not text:
        return text

    sentences = re.split(r"(?<=[.?!])\s+", text)
    unique = []
    for sentence in sentences:
        normalized = sentence.lower().strip()
        if not normalized:
            continue
        if any(normalized in seen.lower() or seen.lower() in normalized
               for seen in unique):
            break
        unique.append(sentence.strip())

    return " ".join(unique)
