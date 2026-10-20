import torch

from emulator_bench import gln_compat as g


def test_jagged_log_softmax_matches_per_segment():
    logits = torch.randn(10)
    ends = torch.tensor([3, 4, 10])
    expected = torch.cat([torch.log_softmax(c, 0) for c in torch.split(logits, [3, 1, 6])])
    assert torch.allclose(g.jagged_log_softmax_forward(logits, ends), expected, atol=1e-6)


def test_jagged_log_softmax_backward_matches_autograd():
    logits = torch.randn(10, requires_grad=True)
    ends = torch.tensor([3, 4, 10])
    out = g.jagged_log_softmax_forward(logits, ends)
    grad = torch.randn(10)
    (out * grad).sum().backward()
    assert torch.allclose(
        g.jagged_log_softmax_backward(out.detach(), grad, ends), logits.grad, atol=1e-6
    )


def test_scatter_ops_match_loops():
    src, index = torch.randn(6, 4), torch.tensor([0, 1, 1, 0, 2, 1])
    groups = [src[index == i] for i in range(3)]
    assert torch.allclose(
        g.scatter_add(src, index, 0, 5)[:3], torch.stack([x.sum(0) for x in groups]), atol=1e-6
    )
    assert torch.allclose(
        g.scatter_mean(src, index, 0, 3), torch.stack([x.mean(0) for x in groups]), atol=1e-6
    )
    assert torch.equal(
        g.scatter_max(src, index, 0, 3)[0], torch.stack([x.max(0).values for x in groups])
    )
    assert torch.equal(
        g.scatter_min(src, index, 0, 3)[0], torch.stack([x.min(0).values for x in groups])
    )


def test_spmm_matches_dense():
    dense = torch.tensor([[0.0, 2.0, 0.0], [1.0, 0.0, 3.0]])
    index = dense.nonzero().t()
    matrix = torch.randn(3, 5)
    assert torch.allclose(g.spmm(index, dense[tuple(index)], 2, matrix), dense @ matrix, atol=1e-6)
