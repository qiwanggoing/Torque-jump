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

    def _reward_projected_peak(self):
        # ONE-SIDED height FLOOR for MAX-DISTANCE (2026-07-09, user). The parent's projected_peak is a BELL
        # centered on the commanded apex -> it penalizes jumping HIGHER than commanded, biasing the launch to
        # a fixed height and away from the distance-optimal angle (symptom: the front legs' push goes into
        # HEIGHT, not reach). Here we FLOOR instead: reward the projected apex UP TO the command (dense
        # up-gradient == the parent below the floor, so JUMP DISCOVERY is preserved), then FLAT at/above it
        # (no reward, no penalty). Height above the floor is FREE -> forward_reach (vx * flight_time) picks the
        # apex that MAXIMIZES DISTANCE on its own. Floor = commands[:,3] (jump_height, set to 0.40 = the min
        # real jump; successful_jump's peak>=0.40 gate agrees). Same gates as the parent.
        base_height = self.root_states[:, 2]
        min_height = float(getattr(self.cfg.rewards, "ascending_min_base_height", 0.18))
        vz = self.root_states[:, 9]
        ascending = (
            self.jumping_state
            & self.has_taken_off
            & (vz > 0)
            & (~self.has_landed)
            & (base_height > min_height)
            & self._squat_deep_enough()
        )
        projected = base_height + torch.clamp(vz, min=0.0) ** 2 / (2.0 * 9.81)
        target = self.commands[:, 3]
        sigma = max(float(getattr(self.cfg.rewards, "projected_peak_sigma", 0.05)), 1e-4)
        shortfall = torch.clamp(target - projected, min=0.0)   # 0 once at/above the floor -> FLAT above
        reward = torch.exp(-torch.square(shortfall) / sigma)
        return ascending.float() * reward

