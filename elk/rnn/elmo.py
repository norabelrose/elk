from allennlp.modules.elmo import Elmo, batch_to_ids
from sacremoses import MosesTokenizer
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class ElmoConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = 1024
        self.num_hidden_layers = 1
        self.is_encoder_decoder = False


class ElmoModel(PreTrainedModel):
    def __init__(self, options_file, weights_file):
        super().__init__(config=ElmoConfig())
        self.elmo_model = Elmo(options_file, weights_file, 1, dropout=0)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
    ):
        return self.elmo_model(input_ids)["elmo_representations"][0]

    @staticmethod
    def from_pretrained(path):
        options_file = (
            "/home/kyle/elk/elk/rnn/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
        )
        weights_file = (
            "/home/kyle/elk/elk/rnn/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
        )
        return ElmoModel(options_file, weights_file)


class ElmoTokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = MosesTokenizer()

    def __call__(self, text, return_tensors, truncation):
        sequences = text if isinstance(text, list) else [text]
        tokens = [
            self.tokenizer.tokenize(sequence, escape=False) for sequence in sequences
        ]
        character_ids = batch_to_ids(tokens)  # type: ignore
        return character_ids

    @staticmethod
    def from_pretrained(path):
        return ElmoTokenizer()
