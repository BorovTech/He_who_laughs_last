import torch
from typing import Type
from torch import nn
from dataset import TextDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions.categorical import Categorical


class LanguageModel(nn.Module):
    def __init__(
        self,
        dataset: TextDataset,
        embed_size: int = 256,
        hidden_size: int = 256,
        rnn_type: Type = nn.RNN,
        rnn_layers: int = 1,
    ):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=embed_size
        )
        self.rnn = rnn_type(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(in_features=hidden_size, out_features=self.vocab_size)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """

        embeds = self.embedding(indices)
        packed_embeds = pack_padded_sequence(
            embeds, lengths, batch_first=True, enforce_sorted=False
        )
        outputs, hidden = self.rnn(packed_embeds)

        outputs, lengths = pad_packed_sequence(outputs, batch_first=True)
        logits = self.linear(outputs)

        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = "", temp: float = 1.0) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Encode the prefix (do not forget the BOS token!),
        pass it through the model to accumulate RNN hidden state and
        generate new tokens sequentially, sampling from categorical distribution,
        until EOS token or reaching self.max_length.
        Do not forget to divide predicted logits by temperature before sampling
        """
        device = next(self.parameters()).device
        tokens = [self.dataset.bos_id]
        tokens += self.dataset.text2ids(prefix)
        tokens = torch.tensor(tokens).unsqueeze(0).to(self.device)

        # generate hidden for prefix
        embeds = self.embedding(tokens)
        output, hidden = self.rnn(embeds)
        logits = self.linear(output)
        logits = logits / temp
        # sample new token from logits
        new_tokens = Categorical(logits=logits[:, -1:]).sample()
        tokens = torch.cat([tokens, new_tokens], dim=1)

        while tokens.shape[1] < self.max_length:
            if new_tokens.item() == self.dataset.eos_id:
                break

            # process newly obtained token
            embeds = self.embedding(new_tokens)
            output, hidden = self.rnn(embeds, hidden)
            logits = self.linear(output)
            # sample the next token from logits
            logits = logits / temp
            new_tokens = Categorical(logits=logits[:, -1:]).sample()
            tokens = torch.cat([tokens, new_tokens], dim=1)

        generated = self.dataset.ids2text(tokens.squeeze())
        return generated

    def to(self, device, **kwargs):
        self.device = device
        return super().to(device, **kwargs)
