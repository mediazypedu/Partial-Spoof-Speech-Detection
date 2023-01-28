import torch
import numpy as np
import random
from longformer.longformer import LongformerSelfAttention, LongformerConfig




def _run_test(attn, hidden_state, attention_mask):
    output3 = attn(hidden_states=hidden_state, attention_mask=attention_mask if attention_mask is not None else None)[0]

    output1 = attn(hidden_states=hidden_state[:1], attention_mask=attention_mask[:1] if attention_mask is not None else None)[0]
    output2 = attn(hidden_states=hidden_state[1:], attention_mask=attention_mask[1:] if attention_mask is not None else None)[0]
    return output3

def test_selfattention():
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    seqlen = 1024
    embed_dim = 60
    num_heads = 3
    bsz = 3
    config = LongformerConfig()
    config.num_attention_heads = num_heads
    config.hidden_size = embed_dim
    config.attention_probs_dropout_prob = 0.0
    config.attention_window = [256]
    config.attention_dilation = [1]
    config.attention_mode = 'sliding_chunks'
    config.autoregressive = False

    attn = LongformerSelfAttention(config=config, layer_id=0)

    hidden_state = torch.randn(bsz, seqlen, embed_dim)
    attention_mask = torch.zeros((bsz, 1, 1, seqlen), dtype=torch.int)  # local attention everywhere

    # test None attention_mask (default which is local attention everywhere)
    output_nonemask = _run_test(attn, hidden_state, None)
    output = _run_test(attn, hidden_state, attention_mask)

    # test padding
    attention_mask[:, :, :, -10:] = -1
    _run_test(attn, hidden_state, attention_mask)

    # test same global attention on all examples
    attention_mask[:, :, :, :10] = 1
    _run_test(attn, hidden_state, attention_mask)

    # test same number of global attention but different locations
    attention_mask[:] = 0
    attention_mask[:, :, :, -10:] = -1
    attention_mask[0, :, :, :10] = 1
    attention_mask[1, :, :, 5:15] = 1
    attention_mask[2, :, :, 10:20] = 1
    _run_test(attn, hidden_state, attention_mask)

    # test variable number of global attention
    attention_mask[:] = 0
    attention_mask[:, :, :, -10:] = -1
    attention_mask[0, :, :, 5:15] = 1
    attention_mask[2, :, :, 13:17] = 1
    _run_test(attn, hidden_state, attention_mask)



if __name__ == '__main__':
    test_selfattention()
