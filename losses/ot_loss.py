import math

import torch


def cosine_cost(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """C_ij = 1 - cos(x_i, y_j) for batched point clouds x: [B, n, d], y: [B, m, d]."""
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-7)
    y = y / y.norm(dim=-1, keepdim=True).clamp_min(1e-7)
    return 1. - x @ y.transpose(-1, -2)


def sinkhorn(cost: torch.Tensor, eps: float = 0.05, n_iters: int = 100, tol: float = 1e-4):
    """Entropic optimal transport between uniform measures, log-domain Sinkhorn.

    Solves  min_P <P, C> + eps * KL(P | a b^T)  s.t.  P 1 = a, P^T 1 = b  for every batch element.

    The potentials are found without autograd. The returned cost is differentiable w.r.t. `cost`
    through the envelope theorem (dOT_eps/dC = P*), so the backward pass is one elementwise
    product instead of an unrolled graph of `n_iters` logsumexps.

    cost: [..., n, m]. Returns (ot_eps [...], plan [..., n, m]).
    """
    n, m = cost.shape[-2:]
    log_a = torch.full(cost.shape[:-1], -math.log(n), device=cost.device, dtype=cost.dtype)
    log_b = torch.full(cost.shape[:-2] + (m,), -math.log(m), device=cost.device, dtype=cost.dtype)

    with torch.no_grad():
        c = cost.detach()
        f = torch.zeros_like(log_a)
        g = torch.zeros_like(log_b)
        for it in range(n_iters):
            f = eps * (log_a - torch.logsumexp((g.unsqueeze(-2) - c) / eps, dim=-1))
            g_new = eps * (log_b - torch.logsumexp((f.unsqueeze(-1) - c) / eps, dim=-2))
            # convergence check every 10 iterations to keep host syncs rare on GPU/MPS
            if it % 10 == 9 and (g_new - g).abs().max() < tol:
                g = g_new
                break
            g = g_new
        plan = torch.exp((f.unsqueeze(-1) + g.unsqueeze(-2) - c) / eps)
        # at the fixed point the dual objective <a,f> + <b,g> equals the entropic OT value
        dual = (log_a.exp() * f).sum(-1) + (log_b.exp() * g).sum(-1)
        linear = (plan * c).sum((-2, -1))

    ot = (plan * cost).sum((-2, -1))
    return ot + (dual - linear), plan


def sinkhorn_divergence(x: torch.Tensor, y: torch.Tensor, eps: float = 0.05, n_iters: int = 100,
                        cost_fn=cosine_cost, yy: torch.Tensor = None) -> torch.Tensor:
    """Debiased Sinkhorn divergence S_eps(x, y) = OT(x,y) - OT(x,x)/2 - OT(y,y)/2 (Feydy et al., 2019).

    The debiasing removes the entropic blur so that S_eps(x, x) = 0 and minimising it does not
    collapse x onto a single point. `yy` may hold a precomputed OT(y, y) when y is constant.
    """
    xy, _ = sinkhorn(cost_fn(x, y), eps, n_iters)
    xx, _ = sinkhorn(cost_fn(x, x), eps, n_iters)
    if yy is None:
        yy, _ = sinkhorn(cost_fn(y, y), eps, n_iters)
    return xy - 0.5 * xx - 0.5 * yy
