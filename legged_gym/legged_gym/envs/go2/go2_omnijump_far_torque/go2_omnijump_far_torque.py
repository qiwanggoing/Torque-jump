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
