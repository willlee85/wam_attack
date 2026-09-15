import pytest
import torch

from attacks.sms.sms import (
    make_fastwam_policy,
    measure_sms,
    pixel_directions,
)


@pytest.mark.parametrize('count', [1, 4, 8, 16])
def test_pixel_directions_are_reproducible_unit_rms_rademacher(count):
    x=torch.zeros(1,3,4,5)
    before=torch.random.get_rng_state().clone()
    first=pixel_directions(x,count,71)
    second=pixel_directions(x,count,71)
    assert torch.equal(first,second)
    assert torch.equal(before,torch.random.get_rng_state())
    assert set(first.unique().tolist())=={-1.,1.}
    torch.testing.assert_close(first.flatten(1).square().mean(1).sqrt(),torch.ones(count))


def test_sms_linear_policy_uses_actual_clipped_pixel_difference():
    x=torch.tensor([[[[-1.,-.2],[.3,1.]]]])
    result=measure_sms(lambda pixels: {'action':2*pixels},x,eta=.1,directions=8,seed=9)
    assert result['valid'] and result['SMS']==pytest.approx(2.,rel=1e-6)
    assert all(row['direction_RMS']==1 for row in result['directions'])
    assert all(row['RMS_delta_x'] < .2 for row in result['directions'])


def test_sms_full_policy_pairs_keep_bound_conditions_fixed():
    calls=[]
    instruction=object();proprio=torch.tensor([3.])
    def policy(pixels):
        calls.append((instruction,proprio.clone()))
        return pixels.mean().reshape(1)+proprio
    result=measure_sms(policy,torch.zeros(1,1,2,2),eta=.01,directions=4,seed=4)
    assert result['valid'] and len(calls)==8
    assert all(call[0] is instruction and torch.equal(call[1],proprio) for call in calls)


def test_fastwam_policy_binds_seed_and_nonvisual_conditions():
    class Model:
        def infer_action(self,**kwargs):
            return {'action':kwargs['input_image']+kwargs['proprio']}
    with pytest.raises(ValueError):
        make_fastwam_policy(Model(),proprio=1)
    policy=make_fastwam_policy(Model(),proprio=2,seed=42)
    torch.testing.assert_close(policy(torch.tensor([1.]))['action'],torch.tensor([3.]))


def test_sms_rejects_unsupported_direction_count():
    with pytest.raises(ValueError):
        measure_sms(lambda x:x,torch.zeros(1),eta=.1,directions=2)
