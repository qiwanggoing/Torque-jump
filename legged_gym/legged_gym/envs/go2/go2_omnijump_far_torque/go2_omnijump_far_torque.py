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

    def _reward_forward_reach(self):
        # Superlinear (quadratic by default) forward reach reward for Far Jump:
        # Eliminates "bunny-hopping" (repeated tiny 10cm hops to farm airborne linear reach scores)
        # by scaling reward as reach_eff ** power. Leaping 1.0m yields 100x the reward of a 0.1m hop.
        pz = self.root_states[:, 2]
        min_h = float(getattr(self.cfg.rewards, "projected_landing_min_height", 0.40))
        active = (self.airborne & (pz > min_h) & self._jump_commanded() & self._squat_deep_enough()).float()
        g = 9.81
        vz = self.root_states[:, 9]
        h_land = self.env_origins[:, 2] + float(self.cfg.rewards.base_height_target)
        disc = torch.clamp(vz * vz + 2.0 * g * (pz - h_land), min=0.0)
        t_land = (vz + torch.sqrt(disc)) / g
        proj_xy = self.root_states[:, :2] + self.root_states[:, 7:9] * t_land.unsqueeze(1)
        reach_vec = proj_xy - self.takeoff_root_xy                      # projected reach from takeoff
        cmd_vec = self.landing_target[:, :2] - self.takeoff_root_xy     # commanded displacement
        cmd_dist = torch.norm(cmd_vec, dim=1)
        cmd_dir = cmd_vec / cmd_dist.clamp(min=1e-3).unsqueeze(1)
        reach_along = (reach_vec * cmd_dir).sum(dim=1)                  # signed reach along command direction
        reach_eff = torch.minimum(reach_along.clamp(min=0.0), cmd_dist)
        power = float(getattr(self.cfg.rewards, "forward_reach_power", 2.0))
        return active * (reach_eff ** power)

