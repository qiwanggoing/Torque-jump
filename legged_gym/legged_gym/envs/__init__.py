# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin


from .base.legged_robot import LeggedRobot

from .go2.go2_torque.go2_torque_config import GO2TorqueCfg, GO2TorqueCfgPPO
from .go2.go2_torque.go2_torque import GO2Torque
from .go2.go2_jump_torque.go2_jump_torque_config import GO2JumpTorqueCfg, GO2JumpTorqueCfgPPO
from .go2.go2_jump_torque.go2_jump_torque import GO2JumpTorque
from .go2.go2_my_jump_torque.go2_my_jump_torque_config import GO2MyJumpTorqueCfg, GO2MyJumpTorqueCfgPPO
from .go2.go2_my_jump_torque.go2_my_jump_torque import GO2MyJumpTorque
from .go2.go2_omninet_torque.go2_omninet_torque_config import GO2OmniNetTorqueCfg, GO2OmniNetTorqueCfgPPO
from .go2.go2_omninet_torque.go2_omninet_torque import GO2OmniNetTorque
from .go2.go2_omnijump_torque.go2_omnijump_torque_config import GO2OmniJumpTorqueCfg, GO2OmniJumpTorqueCfgPPO
from .go2.go2_omnijump_torque.go2_omnijump_torque import GO2OmniJumpTorque
from .go2.go2_omnijump_curriculum_torque.go2_omnijump_curriculum_torque_config import (
    GO2OmniJumpCurriculumTorqueCfg,
    GO2OmniJumpCurriculumTorqueCfgPPO,
)
from .go2.go2_omnijump_curriculum_torque.go2_omnijump_curriculum_torque import GO2OmniJumpCurriculumTorque
from .go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO


import os

from legged_gym.utils.task_registry import task_registry


task_registry.register("go2_torque", GO2Torque, GO2TorqueCfg(), GO2TorqueCfgPPO())
task_registry.register("go2_jump_torque", GO2JumpTorque, GO2JumpTorqueCfg(), GO2JumpTorqueCfgPPO())
task_registry.register("go2_my_jump_torque", GO2MyJumpTorque, GO2MyJumpTorqueCfg(), GO2MyJumpTorqueCfgPPO())
task_registry.register("go2_omninet_torque", GO2OmniNetTorque, GO2OmniNetTorqueCfg(), GO2OmniNetTorqueCfgPPO())
task_registry.register("go2_omnijump_torque", GO2OmniJumpTorque, GO2OmniJumpTorqueCfg(), GO2OmniJumpTorqueCfgPPO())
task_registry.register(
    "go2_omnijump_curriculum_torque",
    GO2OmniJumpCurriculumTorque,
    GO2OmniJumpCurriculumTorqueCfg(),
    GO2OmniJumpCurriculumTorqueCfgPPO(),
)
task_registry.register("go2_rough", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO())
