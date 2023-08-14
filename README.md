# LSTM

A Long-Short Term Memory module trained on a large corpora of labelled tweets.

The model is large, containing just under 30 million trainable parameters, and takes a while to train. As this is being trained on a home laptop, the corpora has been split into 4 datasets, each requiring the model to take a few days to train.
Because of this, at the end of the notebook, there are cells which enable the model and its state dictionary to be saved, and to be trained on another dataset at a later period.

This notebook contains not only the architecture of the LSTM but all of the data science aspects of handling, preprocessing and training a model on a dataset containing over 1.6 million entities.
