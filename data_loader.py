from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch import Tensor


class DataGenerator:
    ENCODING: Dict[str, np.ndarray] = {
                'N': np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
                'A': np.array([1, 0, 0, 0], dtype=np.float32),
                'C': np.array([0, 1, 0, 0], dtype=np.float32),
                'G': np.array([0, 0, 1, 0], dtype=np.float32),
                'T': np.array([0, 0, 0, 1], dtype=np.float32)}

    NUCLEOTIDES: Dict[str, int] = {
                'N': 0,
                'A': 1,
                'C': 2,
                'G': 3,
                'T': 4
    }

    PADDING: str = 'N'
    MAX_SIZE: int = 40

    @staticmethod
    def pad_sequence(sequence: str | List[str], max_size: int = MAX_SIZE) -> str | List[str]:
        if isinstance(sequence, List):
            return [seq.ljust(max_size, DataGenerator.PADDING)[:max_size] for seq in sequence]
        return sequence.ljust(max_size, DataGenerator.PADDING)[:max_size]

    @staticmethod
    def encode_to_tensor(rna_list: List[str]) -> Tensor:
        return torch.tensor([[DataGenerator.NUCLEOTIDES[i] for i in rna] for rna in rna_list]).long()

    @staticmethod
    def convert_to_RNA(sequence: str) -> str:
        return sequence.replace('U', 'T')

    @staticmethod
    def load_RNA_compete(rna_filename: str) -> List[str]:
        with open(rna_filename) as f:
            rna_list: List[str] = [DataGenerator.pad_sequence(DataGenerator.convert_to_RNA(rna.strip())) for rna in f]
        return rna_list

    @staticmethod
    def rna_compete_to_tensor(rna_filename: str) -> Tensor:
        return DataGenerator.encode_to_tensor(DataGenerator.load_RNA_compete(rna_filename))

    @staticmethod
    def one_hot_encoding(seq: Tensor) -> Tensor:
        embedding_matrix = torch.FloatTensor([value for key, value in DataGenerator.ENCODING.items() if key != 'U'])
        return embedding_matrix[seq]


def read_samples(file_path: str, num_of_samples: int) -> List[str]:
    with open(file_path, 'r') as f:
        lines = [f.readline().strip().split()[0] +
                 f.readline().strip().split()[0]
                 for i in range(num_of_samples)]
    return lines


def create_dataset(rbns_files: List[str], max_len: int) -> Tuple[Tensor, Tensor]:
    # Read positive and negative samples.
    negative_dataset = read_samples(rbns_files[0], max_len)
    positive_dataset = read_samples(rbns_files[-1], max_len)

    # Pad sequences.
    negative_samples = DataGenerator.pad_sequence(negative_dataset)
    positive_samples = DataGenerator.pad_sequence(positive_dataset)
    samples = negative_samples + positive_samples

    # Create labels.
    negative_labels = np.zeros(len(negative_samples))
    positive_labels = np.ones(len(positive_samples))
    labels = np.concatenate([negative_labels, positive_labels])

    # Encode data to tensors.
    samples = DataGenerator.encode_to_tensor(samples)
    labels = torch.FloatTensor(labels).reshape(-1, 1)

    indices = np.random.permutation(samples.shape[0])
    return samples[indices], labels[indices]
