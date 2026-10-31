import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from losses.ot_loss import cosine_cost, sinkhorn, sinkhorn_divergence


def exact_ot_uniform(cost):
    # with uniform weights and n == m an optimal plan is a permutation (Birkhoff)
    r, c = linear_sum_assignment(cost)
    return cost[r, c].mean()


def test_sinkhorn_marginals_and_converges_to_exact_ot():
    torch.manual_seed(0)
    x, y = torch.randn(1, 12, 5, dtype=torch.float64), torch.randn(1, 12, 5, dtype=torch.float64)
    cost = torch.cdist(x, y) ** 2
    exact = exact_ot_uniform(cost[0].numpy())
    gaps = []
    for eps in [0.1, 0.03, 0.01]:
        ot, plan = sinkhorn(cost, eps=eps * cost.max().item(), n_iters=20000, tol=1e-12)
        assert torch.allclose(plan.sum(-1), torch.full((1, 12), 1 / 12, dtype=torch.float64), atol=1e-6)
        assert torch.allclose(plan.sum(-2), torch.full((1, 12), 1 / 12, dtype=torch.float64), atol=1e-6)
        gaps.append((plan * cost).sum().item() - exact)
    # transport cost <P, C> of the entropic plan approaches the exact OT cost from above as eps -> 0
    assert gaps[0] > gaps[1] > gaps[2] >= -1e-9
    assert gaps[2] < 0.03 * exact


def test_sinkhorn_gradient_matches_finite_differences():
    torch.manual_seed(0)
    x = torch.randn(2, 6, 4, dtype=torch.float64, requires_grad=True)
    y = torch.randn(2, 9, 4, dtype=torch.float64)
    f = lambda z: sinkhorn(cosine_cost(z, y), eps=0.1, n_iters=2000, tol=1e-13)[0].sum()
    f(x).backward()
    num = torch.zeros_like(x)
    h = 1e-6
    for idx in np.ndindex(*x.shape):
        e = torch.zeros_like(x)
        e[idx] = h
        num[idx] = (f(x.detach() + e) - f(x.detach() - e)) / (2 * h)
    assert torch.allclose(x.grad, num, atol=1e-5)


def test_sinkhorn_divergence_is_zero_on_identical_and_positive_otherwise():
    torch.manual_seed(0)
    x = torch.randn(3, 8, 16)
    assert sinkhorn_divergence(x, x).abs().max() < 1e-5
    assert (sinkhorn_divergence(x, torch.randn(3, 10, 16)) > 0).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple Silicon GPU")
def test_sinkhorn_divergence_mps_matches_cpu():
    torch.manual_seed(0)
    x, y = torch.randn(4, 8, 512), torch.randn(4, 79, 512)
    cpu = sinkhorn_divergence(x, y)
    mps = sinkhorn_divergence(x.to("mps"), y.to("mps")).cpu()
    assert torch.allclose(cpu, mps, atol=1e-4)
