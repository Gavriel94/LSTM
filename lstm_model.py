from torch import nn
import torch
import time 
import wandb

class LSTM_Model(nn.Module):
    """
    LSTM model used to process and predict sentiment of
        a textual input sequence.
    Extracts data from an input sequence via LSTM modules before the 
        classifier determines an output.

    `train()`: used to train the model for a specified number of epochs. 
    `evaluate()`: Forward pass on validation or test data 
        without updating parameters.
    `model_env()` executes both functions.

    Args:
        nn (nn.Module): Base class for PyTorch neural networks.
    """
    def __init__(self, vocab_size, input_dim):
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

        self.embedding = nn.Embedding(vocab_size, input_dim)

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=input_dim*2,
                            num_layers=3,
                            batch_first=True,
                            dropout=0,
                            bidirectional=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim*4, input_dim*2),
            nn.Dropout(0.1),
            nn.Tanh(),
            nn.Linear(input_dim*2, round(input_dim/2)),
            nn.Dropout(0.1),
            nn.Tanh(),
            nn.Linear(round(input_dim/2), 1)
        )
        
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
        
        x, _ = self.lstm(embeddings)
        x = x[torch.arange(x.shape[0]), text_lengths-1, :]
        x = self.classifier(x)
        return x

    def run_training(self, 
                     training, 
                     validation, 
                     testing, 
                     model, 
                     optimizer,
                     scheduler, 
                     criterion, 
                     epochs, 
                     verbose=True,
                     wandb_track=True):
        """
        Wraps the training and evaluation functions in one method.
        At the end of each training loop, validation data is processed.
            Once all epochs are complete, test data is evaluated.

        Args:
            training (DataLoader): DataLoader with training data.
            validation (DataLoader): DataLoader with validation data.
            testing (DataLoader): DataLoader with testing data.
            model (nn.Module): The LSTM model being trained.
            optimizer (torch.optim.Adam): Backpropagation method.
            criterion (torch.nn.modules.loss): Loss function.
            epochs (int): Number of epochs the model is trained for.
            verbose (bool): Display metrics (default=True).
            wandb_track (bool): Track metrics with wandb (default=True).

        Returns:
            train_accuracy, train_loss, val_accuracy, val_loss 
                (list, list, list, list): Metrics saved during training and
                evaluation.
        """
        # Containers for training and evaluation metrics
        train_accuracy = []
        train_loss = []
        val_accuracy = []
        val_loss = []
        # Time saved for calculating final processing time
        start_time = time.time()
        for epoch in range(epochs):
            epoch_start = time.time()
            print('-' * 34)
            print(f'|             Epoch {epoch + 1:<2}           |')
            print('-' * 34)
            # Process training data
            t_loss, t_acc, learning_rate = self.__train(training, 
                                    model, 
                                    optimizer,
                                    scheduler, 
                                    criterion, 
                                    verbose)  
            # Store training metrics
            train_loss.append(t_loss)
            train_accuracy.append(t_acc)
            # Evaluate validation data
            v_loss, v_acc = self.__evaluate(validation, 
                                            model, 
                                            criterion,
                                            epoch_start,
                                            test_data=False)
            # Store evaluation metrics
            val_loss.append(v_loss)
            val_accuracy.append(v_acc)
            # Log metrics to wandb
            
            if wandb_track:
                wandb.log({
                'Epoch': epoch,
                'Training Accuracy': t_acc,
                'Training Loss': t_loss,
                'Validation Accuracy': v_acc, 
                'Validation Loss': v_loss,
                'Learning Rate': learning_rate
                })

        # Assess model performance on test data
        self.__evaluate(testing, 
                        model, 
                        criterion, 
                        start_time, 
                        test_data=True)

        if wandb_track:
                wandb.finish()
        return train_accuracy, train_loss, val_accuracy, val_loss
    
    def __train(self,
            dataloader, 
            model, 
            optimizer, 
            scheduler, 
            criterion, 
            verbose=True):
        """
        Used to train a neural network.
        Iterates through label/text pairs from each dataset making label
        predictions. 
        Calculates loss and backpropagates parameter updates
        through the network to ideally reduce loss over epochs.

        Args:
            dataloader (DataLoader): DataLoader containing training
                data.
            model (nn.Module): The LSTM model being trained.
            optimizer (torch.optim.Adam): Backpropagation method.
            criterion (torch.nn.modules.loss): Loss function.
            verbose (bool): Display metrics (default=True).

        Returns:
            epoch_loss, epoch_accuracy, current_lr 
                (float, float, float): 
                normalised loss, normalised accuracy and current lr.
        """
        model.train()
        # Accuracy and loss accumulated over epoch
        total_accuracy, total_loss = 0, 0
        # Number of predictions
        num_predictions = 0
        # Displays training metrics every quarter of epoch
        intervals = (len(dataloader) / 4).__round__()
        for idx, (label, text, text_lengths) in enumerate(dataloader):
            # Make prediction
            prediction = model(text, text_lengths)
            label = label.unsqueeze(1)
            # Calculate loss
            loss = criterion(prediction, label.float())
            batch_loss = loss.item()
            # Update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Store metrics
            total_accuracy += ((prediction > 0.5) == label).sum().item()
            total_loss += batch_loss
            num_predictions += label.size(0)

            if verbose and idx % intervals == 0 and idx > 0:
                epoch_metrics = (
                    f'| {idx:5} / {len(dataloader):5} batches |' 
                    f' {(total_accuracy/num_predictions)*100:.2f}% |'
                    )
                print(epoch_metrics)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        average_accuracy_pct = (total_accuracy / num_predictions) * 100
        average_loss_per_sample = total_loss / num_predictions
        return average_loss_per_sample, average_accuracy_pct, current_lr

    def __evaluate(self, dataloader, model, criterion, start_time, test_data):
        """
        Used to evaluate model training.
        Works similarly to the training method, allowing the model
        to make predictions on labelled data, however no parameters are
        updated.

        Args:
            datalxoader (DataLoader): Data for evaluation.
            model (nn.Module): The LSTM model being trained.
            criterion (torch.nn.modules.loss): Loss function.
            start)_time (time)

        Returns:
            batch_loss, batch_accuracy, batch_count 
                (float, float): normalised loss and normalised accuracy.
        """
        model.eval()
        total_accuracy = 0
        num_predictions = 0
        total_loss = 0
        with torch.no_grad():
            for label, text, text_length in dataloader:
                prediction = model(text, text_length)
                label = label.unsqueeze(1)
                loss = criterion(prediction, label.float())
                total_accuracy += ((prediction > 0.5) == label).sum().item()
                num_predictions += label.size(0)
                total_loss += loss.item()
                
        average_accuracy_pct = (total_accuracy / num_predictions) * 100
        average_loss_per_sample = total_loss / num_predictions
        
        
        if not test_data:
            print('-' * 34)
            print(f'| Validation Accuracy   : {average_accuracy_pct:.2f}% |')
            print('-' * 34)
            print(f'| Time Elapsed          : {time.time() - start_time:.2f}s |')
        else:    
            print('*' + '-' * 32 + '*')
            print(f'| Test Accuracy         : {average_accuracy_pct:.2f}% |')
            print('*' + '-' * 32 + '*')
            print(f'| Training Time         : {round((time.time() - start_time)/60)}m  |')
        print('-' * 34)
        print()
        return average_loss_per_sample, average_accuracy_pct