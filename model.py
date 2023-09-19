import torch
import torch.nn as nn


class TinyLM(nn.Module):

    def __init__(self, vocab_size: int = 50257, emb_dim: int = 768, block_size: int = 256, n_att_heads: int = 12,
                 n_decoders: int = 12, device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(block_size, emb_dim)
        self.decoders = nn.Sequential(*(TransformerDecoder(emb_dim, block_size, n_att_heads)
                                        for _ in range(n_decoders)))
        self.final_linear = nn.Linear(emb_dim, vocab_size)
        self.layer_norm = nn.LayerNorm(emb_dim)

    def generate(self, context: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        # Context has to have shape (B,T), so (batch, block_size)
        for _ in range(max_new_tokens):
            # Get last block_size tokens,
            # --> this will increasingly shift further to the end when we add the predicted tokens
            context = context[:, -self.block_size:]
            # Get last token of the predicted tokens (remember middle dim is T in B,T,C)
            logits = self(context)[:, -1, :]
            # Sample next token with probabilities predicted by the model
            probabilities = nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            # Append predicted token to the context
            context = torch.cat([context, next_token], dim=1)
        return context

    def forward(self, x):
        token_emb = self.token_emb(x)  # (B,T,C)
        pos_emb = self.pos_emb(torch.arange(self.block_size, device=self.device))  # (T,C)
        # Broadcast pos to token by addition since dim 0 is missing it will be repeated to dim B
        x = token_emb + pos_emb
        x = self.decoders(x)
        x = self.layer_norm(x)
        logits = self.final_linear(x)
        return logits


class TransformerDecoder(nn.Module):

    def __init__(self, emb_dim: int = 768, block_size: int = 256, n_heads: int = 12, dropout: float = 0.2):
        """
        :param emb_dim: Dimension C of the self-attention head = Embedding dimensionality
        :param block_size: Dimension T in the self-attention head = The sequence length of each block of text.
        :param n_heads: Number of self-attention heads
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads

        # Define multi-head masked self-attention
        self.head_projection = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.register_buffer('tril', ~torch.tril(torch.ones(block_size, block_size)).type(torch.bool))
        self.self_attention = nn.MultiheadAttention(emb_dim, n_heads, batch_first=True, dropout=0.2)

        # Define feedforward layers
        self.feed_fwd = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim),
            nn.Dropout(dropout)
        )

        self.ln_1 = nn.LayerNorm(emb_dim)
        self.ln_2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = self.ln_1(x)
        # Input should be of shape B,T,C
        # where B=batch, T=sequence length=block_size, C=embedding dim
        x_proj = self.head_projection(x)
        q, k, v = x_proj.split(self.emb_dim, dim=-1)
        # No need to swap dimensions here as we use batch_first=True in the constructor.
        # Residual connections should help if we would make the network deeper (e.g. more ffwd layers)
        x = x + self.self_attention(q, k, v, attn_mask=self.tril, need_weights=False)[0]
        x = x + self.feed_fwd(self.ln_2(x))
        return x


if __name__ == "__main__":
    t = TinyLM()
    out = t(torch.randn(32, 256, 768))
    print(out.shape)
