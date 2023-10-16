# LSTM

A Long-Short Term Memory module trained on a large corpora of labelled tweets.

The LSTM is an improvement on the Recurrent Neural Network (RNN) model. RNNs perform poorly on input sequences with significant length. In a textual sequence, an RNNs 'memory' would be trying to 'remember' each word in order to deduce meaning from the text. This memory is limited, by the time the RNN is at the end of the sentence, there is a high chance it has 'forgotten' words from the beginning. This is the long-term dependency problem.

LSTMs were introduced to overcome this by carrying a representation of the sequence and updating it by adding/removing things from it. LSTMs are able to capture meaning and sentiment over sequential data without as much information decay.

The model is has around 3 million trainable parameters and takes a while to train. The corpora has been split into 16 datasets so the model can be trained periodically, one subset at a time.

This notebook is an exploration of the LSTM model and also data science aspects of handling, preprocessing and training a model on a large dataset of almost 1.8 million tweets.
