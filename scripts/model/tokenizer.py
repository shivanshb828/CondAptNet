"""
DNA 3-mer tokenizer for CondAptNet.

Maps overlapping 3-mer windows of a DNA sequence to integer token IDs.
All 64 possible DNA 3-mers (4^3) are enumerated deterministically; two
special IDs are reserved: PAD (0) and UNK (1).

Token ID layout:
    0          → [PAD]
    1          → [UNK]  (any 3-mer containing non-ATGC bases)
    2 … 65     → AAA, AAT, AAG, AAC, ATA, ATT, …, CCC  (alphabetically sorted)

Shapes:
    Input  : str  of length L (nucleotides)
    Output : List[int] of length L - 2  (one token per 3-mer window)
    Padded : Tensor [seq_len] with PAD_ID filling up to max_len

Usage:
    tokenizer = DNATokenizer()
    ids = tokenizer.encode("ATGCATGCATGC")
    tensor = tokenizer.encode_padded("ATGCATGCATGC", max_len=50)
"""

import sys
import itertools
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import DNA_KMER_SIZE, DNA_VOCAB_SIZE, DNA_PAD_ID, DNA_UNK_ID, DNA_MAX_LEN, DEVICE


class DNATokenizer:
    """
    Overlapping k-mer tokenizer for native DNA sequences.
    KMER_SIZE=3 by default (64 possible tokens + PAD + UNK = 66 vocab).
    """

    def __init__(self, kmer_size: int = DNA_KMER_SIZE) -> None:
        self.kmer_size = kmer_size
        self.PAD_ID = DNA_PAD_ID
        self.UNK_ID = DNA_UNK_ID

        # Build vocab: sorted list of all k-mers, offset by 2 for PAD/UNK
        bases = ["A", "T", "G", "C"]
        all_kmers = sorted("".join(p) for p in itertools.product(bases, repeat=kmer_size))
        self._kmer_to_id = {km: idx + 2 for idx, km in enumerate(all_kmers)}
        self._id_to_kmer = {v: k for k, v in self._kmer_to_id.items()}
        self._id_to_kmer[self.PAD_ID] = "[PAD]"
        self._id_to_kmer[self.UNK_ID] = "[UNK]"

        expected_vocab = 4 ** kmer_size + 2
        assert len(self._kmer_to_id) + 2 == expected_vocab, "Vocab size mismatch"

    @property
    def vocab_size(self) -> int:
        return 4 ** self.kmer_size + 2

    def encode(self, sequence: str) -> List[int]:
        """
        Encode a DNA string into a list of k-mer token IDs.
        Returns list of length max(0, len(sequence) - kmer_size + 1).
        """
        seq = sequence.strip().upper()
        ids = []
        k = self.kmer_size
        for i in range(len(seq) - k + 1):
            kmer = seq[i : i + k]
            ids.append(self._kmer_to_id.get(kmer, self.UNK_ID))
        return ids

    def encode_padded(self, sequence: str, max_len: int = DNA_MAX_LEN) -> torch.Tensor:
        """
        Encode and pad/truncate to max_len tokens.
        Returns LongTensor of shape [max_len].
        """
        ids = self.encode(sequence)
        token_len = max_len  # number of k-mer tokens for a sequence of max_len nucleotides
        ids = ids[:token_len]                           # truncate
        ids += [self.PAD_ID] * (token_len - len(ids))  # pad
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: List[int]) -> str:
        """
        Reconstruct approximate nucleotide string from token IDs.
        For overlapping k-mers: take first nucleotide of each token + full last token.
        """
        kmers = [self._id_to_kmer.get(i, "?") for i in ids if i not in (self.PAD_ID,)]
        if not kmers:
            return ""
        result = "".join(km[0] for km in kmers[:-1]) + kmers[-1]
        return result

    def batch_encode_padded(self, sequences: List[str], max_len: int = DNA_MAX_LEN) -> torch.Tensor:
        """
        Encode a list of sequences into a batch tensor [batch, max_len].
        """
        tensors = [self.encode_padded(s, max_len) for s in sequences]
        return torch.stack(tensors)


if __name__ == "__main__":
    tokenizer = DNATokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}  (expected {DNA_VOCAB_SIZE})")
    assert tokenizer.vocab_size == DNA_VOCAB_SIZE, "Vocab size mismatch with config.py"

    test_seq = "ATGCATGCATGCATGCATGCATGCAT"
    ids = tokenizer.encode(test_seq)
    print(f"Sequence: {test_seq}")
    print(f"Encoded ({len(ids)} tokens): {ids[:10]}...")

    padded = tokenizer.encode_padded(test_seq, max_len=50)
    assert padded.shape == (50,), f"Wrong shape: {padded.shape}"
    print(f"Padded tensor shape: {padded.shape}")

    decoded = tokenizer.decode(ids)
    assert decoded == test_seq, f"Decode mismatch: {decoded!r} != {test_seq!r}"
    print(f"Decoded: {decoded}")

    # Batch test
    seqs = ["ATGCATGCATGCATGCATGCAT", "GCTAGCTAGCTAGCTAGCTAG"]
    batch = tokenizer.batch_encode_padded(seqs, max_len=30)
    assert batch.shape == (2, 30), f"Wrong batch shape: {batch.shape}"
    print(f"Batch shape: {batch.shape}")

    # UNK test
    ids_unk = tokenizer.encode("ATGRNNATGCATGCATGCATGCAT")
    assert tokenizer.UNK_ID in ids_unk, "UNK token not assigned for invalid bases"
    print(f"UNK test passed ({tokenizer.UNK_ID} present in encoded ids)")

    print("\nAll tokenizer tests passed.")
