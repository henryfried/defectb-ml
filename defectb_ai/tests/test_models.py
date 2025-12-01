import numpy as np
import torch

from defectb_ai.models.network import Conv1DSelfAtten, SelfAttention


def test_self_attention_forward_preserves_sequence_length():
    attn = SelfAttention(input_dim=1, embed_dim=8, num_heads=1, out_dim=1, dropout=0.0)
    x = torch.randn(2, 5)

    out = attn(x)

    assert out.shape == (2, 5)
    assert torch.isfinite(out).all()


def test_conv1d_self_attention_forward_shape():
    model = Conv1DSelfAtten(
        input_size=8,
        conv_layer=[[1, 2, 1, 0]],
        num_heads=1,
        head_dim=2,
        output_dims=[4],
        target_size=2,
        learning_rate=1e-3,
        alpha=0.0,
        dropout=0.0,
        dr=0.99,
    )
    x = torch.randn(3, 8)

    out = model(x)

    assert out.shape == (3, 2)
    assert torch.isfinite(out).all()
