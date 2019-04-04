import torch
import torch.nn as nn

from sotoxic.data_helper.data_transformer import DataTransformer
from sotoxic.data_helper.data_loader import DataLoader
from sotoxic.train.trainer import PyTorchModelTrainer
from sotoxic.config import dataset_config

VOCAB_SIZE = 100000
MAX_SEQUENCE_LENGTH = 300
EMBEDDING_SIZE = 300


print("Loading the dataset")
data_transformer = DataTransformer(max_num_words=VOCAB_SIZE, max_sequence_length=MAX_SEQUENCE_LENGTH, char_level=False)
data_loader = DataLoader()
train_sequences, training_labels, test_sequences = data_transformer.prepare_data()


print("Loading the pre-trained word embedding.")
embeddings_index = data_loader.load_embedding(FASTTEXT_PATH)
embedding_matrix = data_transformer.build_embedding_matrix(embeddings_index)
print("Loaded")

import importlib
import sotoxic.models.pytorch.bgru as bgru
import sotoxic.train.trainer as trn
importlib.reload(bgru)
importlib.reload(trn)

def get_bgru_network():
    embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
    embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
    embedding.weight.requires_grad=False
    return bgru.BayesianGRUClassifier(input_size=EMBEDDING_SIZE, hidden_size=60, embedding=embedding)

trainer = trn.PyTorchModelTrainer(model_stamp="FASTTXT_BGRU_64_64", epoch_num=300, learning_rate=1e-3,
                                  verbose_round=40, shuffle_inputs=False, early_stopping_round=10)

model, best_logloss, best_auc, best_val_pred = trainer.train_folds(X=train_sequences, y=training_labels,
                    fold_count=10, batch_size=256, get_model_func=get_bgru_network, skip_fold=0)                                  

print('Done')