"""Maximum Forward Reach / Far Jump environment on top of the landing-torque infrastructure.

Inherits the validated GO2OmniJumpLandingTorque environment and configures the reward whitelist
and logging to focus on forward reach distance and velocity tracking without landing precision constraints.
"""

import torch
from legged_gym.envs.go2.go2_omnijump_landing_torque.go2_omnijump_landing_torque import (
    GO2OmniJumpLandingTorque,
)
from legged_gym.envs.go2.go2_omnijump_far_torque.go2_omnijump_far_torque_config import (
    GO2OmniJumpFarTorqueCfg,
)


class GO2OmniJumpFarTorque(GO2OmniJumpLandingTorque):
    cfg: GO2OmniJumpFarTorqueCfg

    # Inherit the active reward whitelist and include tracking_linear_velocity
    ACTIVE_REWARD_WHITELIST = GO2OmniJumpLandingTorque.ACTIVE_REWARD_WHITELIST | {
        "tracking_linear_velocity",
    }

    def _init_buffers(self):
        super()._init_buffers()
        # Ensure curriculum is disabled so commands always draw from the full far range
        self.landing_dx_curriculum = False

    # NOTE: the quadratic _reward_forward_reach override was REMOVED (2026-07-08). It scaled reward as
    # reach_eff**2, but reach_eff is in METRES and the achievable range is [0, ~0.7] where x^2 < x -> it
    # SQUASHED the distance reward exactly where the policy operates, and only amplified beyond 1m (which
    # the robot can't reach). Eval proved the result was a VERTICAL HOPPER (~0.13m forward, flat across
    # commands). Reverted to the inherited LINEAR forward_reach (constant gradient toward farther). Bunny-
    # hopping is not a risk here anyway: this is single-jump mode (one jump per episode) + a peak>=0.40
    # height gate, so tiny repeated hops cannot farm the reward.

