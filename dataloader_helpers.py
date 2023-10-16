import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
import pandas as pd


def collate_batch(batch, pipeline, device):
    """
    Custom collate function to prepare batches for LSTM processing.
    Ensures the label is an integer and transforms the input sequence
        to its integer representation.
    
    Args:
        batch (numpy.array): List of tuples containing label/text pairs.
        embeddings (dict): Dictionary of word/int embedding pairs.
        dimensions (int): Dimensions used for embedding vector.
        pipeline (preprocess.PreprocessingPipeline): Data pipeline.
        device (torch.device): Device used to process tensors.

    Returns:
        label_list, text_list, text_lengths 
        (torch.Tensor, torch.Tensor, torch.Tensor): 
            Tensors of labels, padded embeddings and text lengths.
    """
    labels, texts, text_lengths = [], [], []
    for(label, text) in batch:
        l, t = pipeline.label_text_preprocess(label, text)
        labels.append(l)
        texts.append(torch.tensor(t, dtype=torch.int64))
        text_lengths.append(len(t))

    labels = torch.tensor(labels, dtype=torch.int64)
    texts = rnn_utils.pad_sequence(texts, batch_first=True, padding_value=0)
    text_lengths = torch.tensor(text_lengths, dtype=torch.int64)
    return labels.to(device), texts.to(device), text_lengths.to(device)        



def batch_padding(batch_size, pipeline, device):
    """
    Pads batches to specified size then processes using `collate_fn()`. 
    Pads by repeating the last element until batch_size is reached.

    Args:
        batch_size (int): The size of each batch.
    
    Returns:
        collate_fn (function): Function to pad a batch and 
            processes it `collage_batch()`
    """
    def collate_fn(batch):
        padded_batch = batch + [batch[-1]] * (batch_size - len(batch))
        return collate_batch(padded_batch, pipeline, device)
    return collate_fn

def init_dataloaders(folder, 
                     batch_size, 
                     pipeline,
                     device):
    """
    Creates train/val/test DataLoaders.

    Args:
        folder (str): train/val/test csv within 'Data/Split Datasets/'. 
        embedding_dict (dict): word/GloVe embeddings
        batch_size (int): Batch size
        dimensions (int): Dimensions of data. Used for padding batches.
        device (torch.device): Device used to process data.

    Returns:
        train_loader, val_loader, test_loader
            (DataLoader, DataLoader, DataLoader): Required DataLoaders.
    """
    train = pd.read_csv(
        f'Data/Split Datasets/{folder}/train.csv'
        ).drop(columns=['Unnamed: 0'])
    val = pd.read_csv(
        f'Data/Split Datasets/{folder}/val.csv'
        ).drop(columns=['Unnamed: 0'])
    test = pd.read_csv(
        f'Data/Split Datasets/{folder}/test.csv'
        ).drop(columns=['Unnamed: 0'])

    train = train.to_numpy()
    val = val.to_numpy()
    test = test.to_numpy()

    train_loader = DataLoader(
        train, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=batch_padding(batch_size, pipeline, device)
        )
        
    val_loader = DataLoader(
        val, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=batch_padding(batch_size, pipeline, device)
        )

    test_loader = DataLoader(
        test, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=batch_padding(batch_size, pipeline, device)
        )

    return train_loader, val_loader, test_loader