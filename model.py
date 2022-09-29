from statistics import mode
from turtle import forward
import torch
import torch.nn as nn
from torchvision import models
import math

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class Sin_Cos_PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x) :
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

def generate_square_subsequent_mask(size: int):
    """Generate a triangular (size, size) mask."""
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


def find_first(x, element, dim: int = 1):
    """Find the first occurence of element in x along a given dimension.
    Args:
        x: The input tensor to be searched.
        element: The number to look for.
        dim: The dimension to reduce.
    Returns:
        Indices of the first occurence of the element in x. If not found, return the
        length of x along dim.
    Usage:
        >>> first_element(Tensor([[1, 2, 3], [2, 3, 3], [1, 1, 1]]), 3)
        tensor([2, 1, 3])
    Reference:
        https://discuss.pytorch.org/t/first-nonzero-index/24769/9
        I fixed an edge case where the element we are looking for is at index 0. The
        original algorithm will return the length of x instead of 0.
    """
    mask = x == element
    found, indices = ((mask.cumsum(dim) == 1) & mask).max(dim)
    indices[(~found) & (indices == 0)] = x.shape[dim]
    return indices


class Encoder(nn.Module):
    def __init__(self, d_model=128, n_head = 8, dim_feedforward=4080, num_layers = 6) -> None:
        super().__init__()
        resnet = models.resnet34(pretrained = True)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.Cnn_extractor = nn.Sequential(*(list(resnet.children())[:-2] + [nn.Conv2d(512,d_model,1)]))
        transformer_encocder_layer = nn.TransformerEncoderLayer(d_model,n_head,dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encocder_layer, num_layers)
        self.positional_encoding = nn.Parameter(torch.zeros(1, d_model,6,21))

    def forward(self, x):
        x = self.Cnn_extractor(x) #[bn, d_model, w, h]
        bn, d_model, h, w = x.shape
        #add positional encoding
        x = x + self.positional_encoding[:,:,:h,:w]
        x = x.view(bn, d_model, -1) #[bn, d_model, w*h]
        x = x.permute(-1, 0, 1)
        x = self.transformer_encoder(x)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model=128, n_head = 8, dim_feedforward=4080, num_layers = 6, num_class = 1000) -> None:
        super().__init__()
        self.d_model = d_model
        decoder_transformer = nn.TransformerDecoderLayer(d_model,n_head,dim_feedforward)
        self.decoder_transformer = nn.TransformerDecoder(decoder_transformer, num_layers)
        self.embedding = nn.Embedding(num_class, d_model)
        self.sin_cos_positional_encoding = Sin_Cos_PositionalEncoding(d_model)
        self.fc = nn.Linear(d_model, num_class)
    def forward(self, tgt, encoder_embed):
        #tgt.shape [bn, sq]
        tgt = tgt.permute(1, 0)  # (Sq, bn)
        tgt = self.embedding(tgt)*math.sqrt(self.d_model) #(sq, bn, d_model)
        sq, bn, dim = tgt.shape
        tgt = self.sin_cos_positional_encoding(tgt)
        mask = generate_square_subsequent_mask(sq).to(device)
        output = self.decoder_transformer(tgt, encoder_embed, mask)
        output = self.fc(output)
        return output


class OCR_model(nn.Module):
    def __init__(self, conf_encoder, conf_decoder) -> None:
        super().__init__()
        self.encoder = Encoder(*conf_encoder)
        self.decoder = Decoder(*conf_decoder)
        self.max_output_len = 150
        self.sos_index = 1
        self.eos_index = 2
        self.pad_index = 0
    
    def forward(self, img, tgt):
        """
            img: ảnh đầu vào
            tgt: tensor [bn, sq]
            output: tensor [bn, class, sq]
        """
        encode_embed = self.encoder(img)
        output = self.decoder(tgt, encode_embed) #[sq, bn, class]
        return output.permute(1,2,0)
    
    def predict(self, img):
        """Make predctions at inference time.
        Args:
            x: (B, C, H, W). Input images.
        Returns:
            (B, max_output_len) with elements in (0, num_classes - 1).
        """
        B = img.shape[0]
        S = self.max_output_len

        encoded_x = self.encoder(img)  # (Sx, B, E)

        output_indices = torch.full((B, S), self.pad_index).type_as(img).long()
        output_indices[:, 0] = self.sos_index
        has_ended = torch.full((B,), False)

        for Sy in range(1, S):
            y = output_indices[:, :Sy]  # (B, Sy)
            logits = self.decoder(y, encoded_x)  # (Sy, B, num_classes)
            # Select the token with the highest conditional probability
            output = torch.argmax(logits, dim=-1)  # (Sy, B)
            output_indices[:, Sy] = output[-1:]  # Set the last output token

            # Early stopping of prediction loop to speed up prediction
            has_ended |= (output_indices[:, Sy] == self.eos_index).type_as(has_ended)
            if torch.all(has_ended):
                break

        # Set all tokens after end token to be padding
        eos_positions = find_first(output_indices, self.eos_index)
        for i in range(B):
            j = int(eos_positions[i].item()) + 1
            output_indices[i, j:] = self.pad_index

        return output_indices
    

    

conf_encoder = [128, 8, 4080, 6]
conf_decoder = [128, 8, 4080, 6, 1000]

# model = OCR_model(conf_encoder, conf_decoder)
# img = torch.rand(2,3,32,128)
# tgt = torch.tensor([[0,1,2,3],[1,2,3,4]])
# # out = model(img, tgt)
# pred = model.predict(img)


# print(pred)



# model = Encoder()
# x = torch.rand(2,3,32,128)
# print(model(x).shape)
