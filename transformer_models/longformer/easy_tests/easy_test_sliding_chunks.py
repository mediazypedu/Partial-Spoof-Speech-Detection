import time
import torch
import numpy as np
import random
from longformer.sliding_chunks import sliding_chunks_matmul_pv, sliding_chunks_matmul_qk


def same_storage(x, y):
    '''Tests if two tensors share the same underlying storage (for memory optimizations)'''
    return x.storage().data_ptr() == y.storage().data_ptr()

if __name__ == '__main__':
    np.random.seed(3)
    random.seed(3)
    torch.manual_seed(3)
    torch.cuda.manual_seed(3)
    torch.cuda.manual_seed_all(3)

    torch.set_printoptions(sci_mode=False)
    N = 4096  # * 16
    M = 64  # hidden size
    W = 256  # one sided. Actual window size = 2w+1
    B = 3
    D = 1  # no dilation
    H = 12  # number of heads
    autoregressive = False  # not autoregressive
    device = 'cuda'
    dtype = torch.float32

    query = torch.randn(B * N * H * M, requires_grad=True, device=device, dtype=dtype).view(B, N, H, M)
    key = torch.randn(B * N * H * M, requires_grad=True, device=device, dtype=dtype).flip(dims=(0,)).view(B, N, H, M)
    value = torch.randn(B * N * H * M, requires_grad=True, device=device, dtype=dtype).view(B, N, H, M)

    # query = query.half()  # uncomment to profile the fp16 performance
    # key = key.half()
    # value = value.half()
    attention = sliding_chunks_matmul_qk(query, key, W, float('-inf'))
    attention_probs = torch.nn.functional.softmax(attention, dim=-1)
    context2 = sliding_chunks_matmul_pv(attention_probs, value, W)
