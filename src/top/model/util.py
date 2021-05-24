#!/usr/bin/env python3
"""General utility functions concerned with tensor operations in torch."""

import torch as th

#def strides(shape:Tuple[int,...]):
#    shape = th.as_tensor(shape) # e.g. [2,3,5]
#    shape_flip 
#    out   = th.ones(len(shape), dtype = th.int32)
#    th.cumprod(shape, dim=0, out = out[:-1]) # e.g. [2,6,30
#    out = th.flip(out, dims = (0,))
#
#def flat_spatial_index(index:th.Tensor, shape:Tuple[int,...], 
#        normalized:bool = True, dim:int=-1):
#    # index : [..., |shape|]
#    shape   = th.as_tensor(shape,dtype=th.int32,device=index.device)
#    stride  = strides(shape)
#    if normalized:
#        index = index * shape
#    min_ok = th.all(index >= 0, dim = dim)
#    max_ok = th.all(index < shape, dim=dim)
#    mask   = th.logical_and(min_ok, max_ok)
#
#def spatial_gather(x: th.Tensor, ind: th.Tensor, mask: th.Tensor = None):
#    """Gather along spatial axes from a tensor of shape (N,C,...).
#
#    NOTE(ycho): taken from `CenterNet`, initially called "_gather_feat".
#    `ind` is expected to be flat index, e.g. expected output from
#    np.ravel_multi_index.
#    """
#    dim = x.size(2)  # x = [BATCH X NUM_OBJECTS X FEAT(?)]
#    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
#    x = x.gather(1, ind)
#    if mask is not None:
#        mask = mask.unsqueeze(2).expand_as(x)
#        x = x[mask]
#        x = x.view(-1, dim)
#    return x
