"""Backend contract shared by the i.MX8MP and Hailo-8 Whisper implementations.

Both backends run the *same* model (Whisper-tiny) over the *same* 10 s audio
window, so encoder passes and decoder steps are directly comparable numbers.
"""

import numpy as np

from . import tokenizer as tok


class Backend:
    """A Whisper-tiny encoder/decoder pair running on one accelerator."""

    #: Human readable name shown in the report.
    name = "backend"
    #: What the model actually executes on.
    device = "unknown"
    #: Seconds of audio consumed by one encoder pass.
    chunk_length = 10
    #: Number of decoder positions the compiled decoder graph holds.
    sequence_length = 0

    def encode(self, mel):
        """Run one encoder pass. *mel* is [n_mels, frames] float32."""
        raise NotImplementedError

    def decode_logits(self, encoded, token_ids, position):
        """Run one decoder pass and return the logits at *position*.

        *token_ids* is the full fixed-length decoder input sequence; the
        decoder graphs of both backends are non-cached, so every generated
        token costs a complete decoder pass on both devices.
        """
        raise NotImplementedError

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- shared greedy decoding loop -------------------------------------

    def generate(self, encoded, max_tokens=None, no_repeat_ngram=4):
        """Greedy-decode a transcript from an encoded window.

        Identical for both backends, so the only thing the timing difference
        measures is the cost of decode_logits().

        Generation ends on the end-of-transcript token, on the first repeated
        *no_repeat_ngram*-gram, or when the decoder's fixed sequence is full.
        The n-gram guard matters here: a 10 s window usually cuts an utterance
        mid-sentence, and greedy Whisper-tiny then loops on the trailing words
        forever instead of emitting EOT.

        Returns (tokens, stop_reason).
        """
        seq_len = self.sequence_length
        limit = seq_len - 1 if max_tokens is None else min(max_tokens, seq_len - 1)

        token_ids = np.zeros((1, seq_len), dtype=np.int64)
        for index, token in enumerate(tok.FORCED_DECODER_IDS):
            if index < seq_len:
                token_ids[0][index] = token

        # Free generation starts on the last forced token.
        first = len(tok.FORCED_DECODER_IDS) - 1
        generated = []
        seen_ngrams = set()
        stop_reason = "sequence-full"

        for position in range(first, limit):
            logits = self.decode_logits(encoded, token_ids, position)
            next_token = int(np.argmax(logits))

            if next_token == tok.EOT_TOKEN:
                stop_reason = "end-of-transcript"
                break

            generated.append(next_token)
            token_ids[0][position + 1] = next_token

            if no_repeat_ngram and len(generated) >= no_repeat_ngram:
                ngram = tuple(generated[-no_repeat_ngram:])
                if ngram in seen_ngrams:
                    del generated[-no_repeat_ngram:]
                    stop_reason = f"repeated {no_repeat_ngram}-gram"
                    break
                seen_ngrams.add(ngram)

        return generated, stop_reason
