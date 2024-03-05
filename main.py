import time
from typing import List
import torch
import numpy as np
import argparse
from train_model import train_model
from data_loader import DataGenerator
from torch import Tensor
from torch.nn import Module


def predict_rna_compete(model: Module, rna: Tensor, batch_size: int = 64) -> Tensor:
    model.eval()

    with torch.no_grad():
        batched_predictions = [model(DataGenerator.one_hot_encoding(rna[i:i + batch_size]))
                               for i in range(0, rna.shape[0], batch_size)]

    return torch.cat(batched_predictions)


def train_and_predict(pred_path: str, rna: Tensor, rbns_files: List[str]):
    print('Begin training on given RBNS')

    start_time = time.time()
    model = train_model(rbns_files)
    print(r'Training time took {} seconds.'.format(time.time() - start_time))

    start_time = time.time()
    predictions = predict_rna_compete(model, rna)
    print(r'Prediction time took {} seconds.'.format(time.time() - start_time))

    np.savetxt(pred_path, predictions.numpy())
    print('Finished training and predicting for this protein')


parser = argparse.ArgumentParser(
    description="Deep learning BIO project\n You are required to run the code with at "
                "least 7 parameters:\n <ofile> <RNCMPT> <input> <RBSN1> <RBSN2> ... <RBSN5>"
)

parser.add_argument('ofile', type=str, help='Path of ofile')
parser.add_argument('RNCMPT', type=str, help='Path of RNAcompete filename')
parser.add_argument('RBNS_files', nargs='+', type=str, help='File names of RNBS files')

args = parser.parse_args()

if __name__ == '__main__':
    rna_seqs: Tensor = DataGenerator.rna_compete_to_tensor(args.RNCMPT)
    train_and_predict(args.ofile, rna_seqs, args.RBNS_files)
