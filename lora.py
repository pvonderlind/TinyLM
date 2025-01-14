import torch
import torch.nn as nn


# For nn.MultiHeadAttention, the following holds:
# k=d=embed_dim , weights are dxd (obviously for self-attention)

# Weight matrix used for TinyLM's MultiHeadAttention modules
# are saved in self.in_proj_weight which is (3*embed_dim, embed_dim) for Q,K,V as they use the same input tensor.
# -> Add the spectral decomposition for self.in_proj_weight -> B = (EMBED_DIM, RANK), A = (RANK, EMBED_DIM)

class AttentionLoRA(nn.Module):

    def __init__(self, module: nn.MultiheadAttention, alpha: float, rank: int = 1,
                 device: str = 'cuda'):
        """
        :param module: The network to finetune
        :param alpha: Scaling parameter alpha
        :param rank: Rank for the output of A and input of B
        """
        super().__init__()

        # Set dimension d
        self.module = module
        self.target_parameter = 'in_proj_weight'
        self.qkv_size = 3

        self.alpha = alpha
        self.rank = rank
        self.scaling = alpha / rank

        self.in_features = module.in_proj_weight.shape[0] // self.qkv_size
        self.out_features = module.in_proj_weight.shape[1]

        # Network has parameters of dimension dxd (d = embed_dim)
        # B has dxr, A has rxd
        # --> For Self-Attention we have more parameters, as nn.MultiHeadAttention contains q,k,v concat
        # in one tensor.
        # --> This is also relevant for the initialization, as we need to init in thirds and not over the
        # whole tensor at once.

        # Initialize B with zeros.
        self.B = nn.Parameter(torch.zeros(self.in_features * self.qkv_size, self.rank, device=device),
                              requires_grad=True)

        # Init A with N(0,sigma^2) -> Kaiming Normal Distribution
        self.A = nn.Parameter(torch.empty(rank, self.out_features, device=device),
                              requires_grad=True)
        self._init_A()

    def _init_A(self):
        # Initialize tensors with Kaiming-Normal over each part of the A tensor that represents all of Q,K,V
        for idx in range(self.qkv_size):
            nn.init.kaiming_normal_(self.A[:, idx * self.out_features: (idx + 1) * self.out_features])

    def _merge_weights(self):
        """This is required to get the benefit of not manually re-writing the forward pass for MultiHeadAttention"""
        # Important to detach, so we don't actually modify the parameters.

        # Shape: (EMBED_DIM * 3, EMBED_DIM)
        attention_weights_det = self.module.get_parameter(self.target_parameter).detach()
        # Shape: (EMBED_DIM * 3, EMBED_DIM)
        merged_ba = (self.B @ self.A).view(attention_weights_det.shape)
        # Shape: (EMBED_DIM * 3, EMBED_DIM)
        merged_weights = attention_weights_det + merged_ba * self.scaling
        # Update the modified parameters
        setattr(self.module, self.target_parameter, nn.Parameter(merged_weights))

    def _un_merge_weights(self):
        target_weights = self.module.get_parameter(self.target_parameter)
        # Update the target parameter by subtracting the LorA weights
        eval(f'self.module.{self.target_parameter}').data -= (self.B @ self.A).view(target_weights.shape) * self.scaling

    def forward(self, *args, **kwargs):
        # Weights of module and LoRA need to be merged before the forward pass!
        # --> Add them together with scaling.
        # Then proceed to normal forward pass.

        self._merge_weights()
        result = self.module(*args, **kwargs)
        self._un_merge_weights()
        return result


def inject_lora(module: nn.Module, target_layers: list[str], rank: int, alpha: float, device: str = 'cuda'):
    mod_layers = []

    for n, m in module.named_modules():
        # Freeze network
        for p in module.parameters():
            p.requires_grad = False
        splits = n.split(".")
        target = splits[-1]
        m.requires_grad = False

        # Record unfrozen LoRA layer if target is found.
        if target in target_layers:
            mod_layers.append((splits, AttentionLoRA(m, alpha, rank, device)))

    for name_lst, lora in mod_layers:
        target = name_lst[-1]
        current_mod = module
        for part in name_lst[:-1]:
            if len(part) == 1:  # int index
                current_mod = current_mod[int(part)]
            else:  # attribute
                current_mod = getattr(current_mod, part)

        setattr(current_mod, target, lora)

# EXAMPLE USAGE:
# 1. LOAD MODEL
# test_model.load_state_dict(torch.load('tiny_lm/tiny_lm_weights')['model_state_dict'])
# 2. INJECT LORA LAYERS AND FREEZE ORIGINAL WEIGHTS
# inject_lora(test_model, ["self_attention"], 2, 0.1, test_model.device)
# 3. Train on fine-tune dataset.
