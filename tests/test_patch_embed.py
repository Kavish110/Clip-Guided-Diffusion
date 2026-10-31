import torch

from losses.clip_loss import PatchEmbedMatmul


def test_patch_embed_matmul_matches_conv_forward_and_gradients():
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(3, 32, kernel_size=16, stride=16, bias=False).double()
    x = torch.randn(2, 3, 64, 64, dtype=torch.float64, requires_grad=True)
    ref = conv(x)
    gx_ref, gw_ref = torch.autograd.grad(ref.square().sum(), (x, conv.weight))

    patch = PatchEmbedMatmul(conv)
    out = patch(x)
    gx, gw = torch.autograd.grad(out.square().sum(), (x, conv.weight))

    assert out.shape == ref.shape
    assert torch.allclose(out, ref)
    assert torch.allclose(gx, gx_ref)
    assert torch.allclose(gw, gw_ref)
