import torch
import torch.nn as nn
from einops import reduce

from .modules import ETSEmbedding
from .encoder import EncoderLayer, Encoder
from .decoder import DecoderLayer, Decoder


class Transform:
    def __init__(self, sigma):
        self.sigma = sigma

    @torch.no_grad()
    def transform(self, x):
        return self.jitter(self.shift(self.scale(x)))

    def jitter(self, x):
        return x + (torch.randn(x.shape).to(x.device) * self.sigma)

    def scale(self, x):
        return x * (torch.randn(x.size(-1)).to(x.device) * self.sigma + 1)

    def shift(self, x):
        return x + (torch.randn(x.size(-1)).to(x.device) * self.sigma)


class ETSformer(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.e = nn.Parameter(torch.randn(1))
        self.d = nn.Parameter(torch.randn(1))
        self.s = nn.Parameter(torch.randn(1))
        self.configs = configs

        assert configs.e_layers == configs.d_layers, "Encoder and decoder layers must be equal"

        self.enc_embedding = ETSEmbedding(configs.enc_in, configs.d_model, dropout=configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    self.configs.d_model, self.configs.n_heads, self.configs.c_out, self.configs.seq_len, self.configs.pred_len,
                    dim_feedforward=self.configs.d_ff,
                    dropout=self.configs.dropout,
                    activation=self.configs.activation,
                    output_attention=self.configs.output_attention,
                ) for _ in range(self.configs.e_layers)
            ]
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    self.configs.d_model, self.configs.n_heads, self.configs.c_out, self.configs.pred_len,
                    dropout=configs.dropout,
                    output_attention=self.configs.output_attention,
                ) for _ in range(self.configs.d_layers)
            ]
        )

        self.transform = Transform(sigma=self.configs.std)

    def forward(self, x_enc, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, decomposed=False, attention=True):
        res = self.enc_embedding(x_enc)

        level, growths, growth_attns = self.encoder(res, x_enc, attn_mask=enc_self_mask)

        growth, growth_dampings = self.decoder(growths)

        if decomposed:
            return level[:, -1:], growth

        preds = level[:, -1:] + growth
        if attention:
            decoder_growth_attns = []
            for growth_attn, growth_damping in zip(growth_attns, growth_dampings):
                decoder_growth_attns.append(torch.einsum('bth,oh->bhot', [growth_attn.squeeze(-1), growth_damping]))

            decoder_growth_attns = torch.stack(decoder_growth_attns, dim=0)[:, :, -self.pred_len:]
            decoder_growth_attns = reduce(decoder_growth_attns, 'l b d o t -> b o t', reduction='mean')
            return preds,  decoder_growth_attns

        return preds, level, growths[0]