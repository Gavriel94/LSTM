from torchtext.data.utils import get_tokenizer

class PreprocessingPipeline:
    """
    Transforms data to a format which can be interpreted by the model.

    Attributes:
        vocab (Vocab): A Vocab object that maps tokens to integers.
        tokenizer (callable): Function which tokenizes a string.
    
    Args:
        vocab (Vocab): Torchtext Vocab object.
    """
    def __init__(self, vocab):
        self.vocab = vocab
        self.tokenizer = get_tokenizer('basic_english')
    
    def label_text_preprocess(self, label, text):
        """
        Transforms label and text for model input.
        
        Ensures labels are integers.
        Converts each token of a string to its integer representation
            based on the vocabulary.

        Args:
            label (int): Label determining output class of string. 
            text (str): String for conversion.

        Returns:
            Tuple(int, List[int]): Transformed label and text.
        """
        label_ = lambda x: int(x)
        text_ = lambda x: [self.vocab[token] for token in self.tokenizer(x)]
        return label_(label), text_(text)