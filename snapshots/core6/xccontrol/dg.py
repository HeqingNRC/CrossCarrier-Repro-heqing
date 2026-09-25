"""Explicit extensions. CosineHeadFishr is an adaptation, not unchanged official Fishr.

SWAD uses the user's vendored LossValley selection; only trainable parameters
are snapshotted. Its source-validation decision metric is supplied by the runner.
"""
from __future__ import annotations
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


def per_sample_cosine_head_grads(z, weights, logits_adjusted, y, scale, smoothing):
    """Exact CE gradient w.r.t. each normalized cosine classifier weight matrix.

For each head: s(p-q) * [zhat - cos(zhat,What) What] / ||W||.
Margin/log-prior are constants wrt W. Softmax is over the *sum* of the two
heads, exactly as in the source trainer. Do not detach: the variance penalty
must backpropagate into features/adapters as well as classification weights.
    """
    zhat=F.normalize(z,dim=1)
    q=F.one_hot(y,logits_adjusted.shape[1]).to(logits_adjusted.dtype)
    q=(1.0-smoothing)*q+smoothing/logits_adjusted.shape[1]
    factors=scale*(F.softmax(logits_adjusted,dim=1)-q)
    out=OrderedDict()
    for name,w in weights.items():
        norm=w.norm(dim=1).clamp_min(1e-12)
        what=w/norm[:,None]
        cosine=zhat@what.t()
        tangent=zhat[:,None,:]-cosine[:,:,None]*what[None,:,:]
        out[name]=(factors[:,:,None]*tangent/norm[None,:,None]).flatten(1)
    return out


class CosineHeadFishr:
    def __init__(self,num_domains,ema=0.95):
        if not 0<=ema<1: raise ValueError('Fishr ema must be in [0,1).')
        self.num_domains=num_domains; self.ema=ema
        self.previous=[{} for _ in range(num_domains)]
        self.skipped=0; self.updates=0

    def penalty(self,grads,domains):
        counts=[int((domains==d).sum()) for d in range(self.num_domains)]
        # Variance from <2 examples would be meaningless. Keep baseline sampler
        # unchanged; skip and report such batches, rather than silently resample.
        if min(counts)<2:
            self.skipped+=1
            return next(iter(grads.values())).new_zeros(()),counts
        vectors=[]
        for d in range(self.num_domains):
            parts=[]
            for name,g in grads.items():
                values=g[domains==d]
                var=(values-values.mean(0,keepdim=True)).pow(2).mean(0)
                prev=self.previous[d].get(name,torch.zeros_like(var))
                average=self.ema*prev+(1-self.ema)*var
                # Same one-minus-ema gradient correction as the user's Fishr utilities.
                parts.append((average/(1-self.ema)).flatten())
                self.previous[d][name]=average.detach().clone()
            vectors.append(torch.cat(parts))
        self.updates+=1
        stack=torch.stack(vectors); mean=stack.mean(0)
        return (stack-mean).pow(2).mean(),counts


class ParameterView(nn.Module):
    """Live view; AveragedModel deep-copies this, never the 300M frozen backbone."""
    def __init__(self,model):
        super().__init__()
        named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        self.names=tuple(n for n,p in named)
        self.values=nn.ParameterList(p for n,p in named)


def snapshot_dict(averaged):
    module=averaged.module if hasattr(averaged,'module') else averaged
    return {name:p.detach().cpu().clone() for name,p in zip(module.names,module.values)}
