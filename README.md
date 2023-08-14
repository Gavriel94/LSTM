# LSTM

A Long-Short Term Memory module trained on a large corpora of labelled tweets.

The LSTM is an improvement on the Recurrent Neural Network (RNN) model. RNNs perform poorly on input sequences with significant length. In a textual sequence, an RNNs 'memory' would be trying to 'remember' each word in order to deduce meaning from the text. This memory is limited, by the time the RNN is at the end of the sentence, there is a high chance it has 'forgotten' words from the beginning. This is the long-term dependency problem.

LSTMs were introduced to overcome this by carrying a representation of the sequence and updating it by adding/removing things from it. LSTMs are able to capture meaning and sentiment over sequential data without as much information decay.

The model is large, containing just under 30 million trainable parameters, and takes a while to train. As this is being trained on a home laptop, the corpora has been split into 4 datasets, each requiring the model to take a few days to train.
Because of this, at the end of the notebook, there are cells which enable the model and its state dictionary to be saved, enabling the futher training at a later period.

This notebook contains not only the architecture of the LSTM but also the data science aspects of handling, preprocessing and training a model on a dataset containing over 1.6 million entities.
