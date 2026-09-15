import pytest
import torch

from attacks.sms.objective import differentiable_sms_loss, rademacher_directions


def test_directions_are_reproducible_unit_rms_and_use_private_rng():
    x = torch.zeros(1, 3, 4, 5)
    state = torch.random.get_rng_state().clone()
    first = rademacher_directions(x, 4, 71)
    second = rademacher_directions(x, 4, 71)
    assert torch.equal(first, second)
    assert torch.equal(state, torch.random.get_rng_state())
    torch.testing.assert_close(first.square().flatten(1).mean(1), torch.ones(4))


def test_linear_policy_has_exact_squared_sms():
    center = torch.zeros(1, 1, 2, 2, requires_grad=True)
    directions = rademacher_directions(center, 4, 3)
    result = differentiable_sms_loss(lambda x: 2.5 * x, center, directions, eta=0.1)
    assert result.sms.item() == pytest.approx(2.5, rel=1e-6)
    assert result.loss.item() == pytest.approx(2.5**2, rel=1e-6)
    assert result.loss.requires_grad


def test_nonlinear_sms_gradient_matches_finite_difference():
    center = torch.tensor([0.2, -0.3], dtype=torch.float64, requires_grad=True)
    directions = torch.tensor([[1.0, -1.0]], dtype=torch.float64)

    def policy(x):
        return x + 0.4 * x.square()

    def value(x):
        return differentiable_sms_loss(policy, x, directions, eta=1e-2).loss

    analytic = torch.autograd.grad(value(center), center)[0]
    # The production loss intentionally forms action differences in FP32, so a
    # 1e-3 outer finite-difference step avoids testing below FP32 resolution.
    step = 1e-3
    numerical = []
    for index in range(center.numel()):
        offset = torch.zeros_like(center)
        offset[index] = step
        numerical.append((value(center.detach() + offset) - value(center.detach() - offset)) / (2 * step))
    numerical = torch.stack(numerical).to(analytic.dtype)
    torch.testing.assert_close(analytic, numerical, rtol=2e-3, atol=2e-3)


def test_loss_is_mean_of_direction_squares_not_square_of_mean():
    center = torch.tensor([0.25, -0.5], requires_grad=True)
    directions = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    result = differentiable_sms_loss(lambda x: x.sum().square().reshape(1), center, directions, eta=0.05)
    assert float(result.loss) == pytest.approx(float(result.per_direction_sms.square().mean()))
    assert not torch.isclose(result.loss, result.sms.square())
