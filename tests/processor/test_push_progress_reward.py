import torch

from lerobot.lerobot_types import TransitionKey
from lerobot.processor import PushProgressRewardProcessorStep, create_transition


def make_transition(image: torch.Tensor) -> dict:
    return create_transition(
        observation={"observation.images.top": image},
        reward=0.0,
        done=False,
        truncated=False,
        info={},
        complementary_data={},
    )


def test_push_progress_reward_is_monotonic_and_terminates_on_success():
    processor = PushProgressRewardProcessorStep(
        target_roi=(0, 0, 4, 4),
        success_threshold=80.0,
        progress_reward=0.5,
        success_reward=1.0,
    )
    baseline = torch.zeros(3, 8, 8)

    transition = processor(make_transition(baseline))
    assert transition[TransitionKey.REWARD] == 0.0
    assert transition[TransitionKey.DONE] is False

    halfway = baseline.clone()
    halfway[:, :4, :4] = 40 / 255
    transition = processor(make_transition(halfway))
    assert transition[TransitionKey.REWARD] == 0.25
    assert transition[TransitionKey.DONE] is False

    # Moving back does not create another reward opportunity.
    transition = processor(make_transition(baseline))
    assert transition[TransitionKey.REWARD] == 0.0

    success = baseline.clone()
    success[:, :4, :4] = 80 / 255
    transition = processor(make_transition(success))
    assert transition[TransitionKey.REWARD] == 1.25
    assert transition[TransitionKey.DONE] is True
