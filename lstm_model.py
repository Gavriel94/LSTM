from torch import nn
import torch

class LSTM_Model(nn.Module):
    """
    LSTM model used to process and predict sentiment of
        a textual input sequence.
    Extracts data from an input sequence via LSTM modules before the 
        classifier determines an output.

    Args:
        nn (nn.Module): Base class for PyTorch neural networks.
    """
    def __init__(self, vocab_size, vector_dim, num_hidden_nodes, hidden_layers):
        """
        Initialises LSTM modules, the classifier, 
            batch normalisation, ReLU activation and dropout.

        Args:
            vocab_size (int): Size of the vocabulary. 
                Used to determine size of embedding layer.
            vector_dim (int): Dimensions of vector embeddings.
            num_hidden_nodes (int): Number of nodes in the LSTM module.
            hidden_layers (int): Number of hidden layers in the module.
        """
        super(LSTM_Model, self).__init__()

        self.embedding = nn.Embedding(vocab_size, vector_dim)

        self.lstm1 = nn.LSTM(vector_dim,
                    num_hidden_nodes*2,
                    hidden_layers,
                    bidirectional=False,
                    dropout=0,
                    batch_first=True)

        self.lstm2 = nn.LSTM(num_hidden_nodes*2,
                    num_hidden_nodes*4,
                    hidden_layers,
                    bidirectional=False,
                    dropout=0,
                    batch_first=True)
        
        self.linear1 = nn.Linear(num_hidden_nodes*4, 32)
        self.linear1_batch_norm = nn.BatchNorm1d(32)

        self.linear2 = nn.Linear(32, 16)
        self.linear2_batch_norm = nn.BatchNorm1d(16)
        
        self.linear3 = nn.Linear(16, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, text, text_lengths):
        """
        Determines the order of operations in the model.
        Processes the text through LSTM modules and a classifier.

        Args:
            text (torch.Tensor): List of data.
            text_lengths (torch.Tensor): Tensor representing the length 
                of each text sequence in the batch. 
                Shape: [batch_size]

        Returns:
            x (torch.Tensor): The model's predictions. 
        """
        embeddings = self.embedding(text)
        x, _ = self.lstm1(embeddings)
        x, _ = self.lstm2(x)
        x = x[torch.arange(x.shape[0]), text_lengths-1, :]
        x = self.relu(self.dropout(self.linear1_batch_norm(self.linear1(x))))
        x = self.relu(self.dropout(self.linear2_batch_norm(self.linear2(x))))
        x = self.linear3(x)
        return x 