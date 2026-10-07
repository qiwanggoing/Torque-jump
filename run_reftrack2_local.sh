#!/bin/bash
# "+tracking reward" arm, TAKE 2 -- LOCAL 3060 DIRECTION CHECK, NOT comparison data.
#
# WHY TAKE 2: take 1 (Oct07_21-09-14_reftrack_s1, REF_BASE_H_W=12 / REF_ORI_W=12 / REF_JOINT_W=8
# and NO velocity term) learned to jump IN PLACE. At iters 1000-1600 it matched the control on
# peak_height_error (0.029) and jump_landing_rate (0.992) but had clean_reach 0.138 vs 0.416 and
# landing_hit_rate 0.000 vs 0.157 -- while earning the HIGHEST mean_reward of the three runs.
# Root cause measured on the reference itself (across-command spread / over-jump range):
#   vx 0.294   vy 0.378   pitch 0.211   roll 0.335   q 0.121   z 0.077   vz 0.059
# and in absolute terms takeoff vx spans 0.863 -> 1.903 m/s over cmd 0.30 -> 0.90 while z spans
# 3.5 cm. So the tracked set (z, q, pitch, roll) is almost command-INVARIANT and is fully
# satisfiable by a vertical jump with the right crouch-extend-attitude profile, which is cheaper
# than travelling -> the tracking reward COMPETED with distance instead of guiding it.
# FIX: add ref_base_vel (tracks the recorded world-frame vx,vy,vz = the command-carrying quantity)
# and cut the other three ~6x so the set guides instead of dominating.
#
# TWO DEVIATIONS from sstill6 / refprev2, both because this runs on the local 3060:
#   1. GPU: memory/project_env_stack_confound -- 3060 vs 4090 is a PROVEN systematic gap
#      (3060 air ~0.51 vs 4090 0.72-0.80 after ruling out seed/config/num_envs/torch/.local).
#   2. num_envs 4096 -> 2048: the 3060 Laptop has 6144 MiB total and the 4096-env run takes
#      5663 MiB. This also halves the per-iteration batch.
# So this run answers ONE question only: did adding the velocity term stop the jump-in-place?
# Read it at iters 1000-1500 against take 1's clean_reach 0.138 / landing_hit_rate 0.000.
# The comparable arm has to be re-run on the 4090 once GPU0 frees up.
set -e
cd /home/qiwang/torque_jump2/SATA/legged_gym
export PATH=/home/qiwang/miniconda3/envs/sata-clean/bin:$PATH
export LD_LIBRARY_PATH=/home/qiwang/miniconda3/envs/sata-clean/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1

# ---- sstill6 env block, verbatim (programmatically diffed) ----
export GROWTH_WARMUP=96000 GROWTH_X0=192000 PD_FLOOR=0.2
export DX_MIN=0.3 DX_FLOOR_CURR=0 STAND_ANCHOR=1 STANCE_CURR=0
export PEAK_W=20 SUCC_LAND_SIGMA=0.15 SUCCDIST_CAP=1
export LOAD_FADE=1 PD_FADE_SCOPE=all LOAD_FADE_TIMED=1
export LOAD_FADE_STEP=0.1 LOAD_FADE_HOLD=12000 LOAD_FADE_GUARD=0.6
export PD_TARGET_OBS=1
export STAND_POSE_W=3.0
export STAND_STILL_W=6.0
# ---- the tracking rewards. NO REF_PREVIEW_OBS: this arm changes the reward only. ----
export REF_VEL_W=6.0          # the command-carrying term, highest weight
export REF_BASE_H_W=2.0       # was 12 -- near command-invariant, demoted to posture guidance
export REF_ORI_W=2.0          # was 12 -- same
export REF_JOINT_W=1.3        # was 8  -- the literature's lowest-weighted term anyway
export REF_SIG_H=0.05         # literature is 0.02, but that is a dead kernel from scratch (see config)
# REF_SIG_VEL defaults to 0.30; REF_SIG_JOINT 0.2 / _LAND 2.0 / REF_SIG_ORI 0.10 are the literature values.

exec python legged_gym/scripts/train.py --task=go2_omnijump_landing_torque \
  --headless --seed 1 --max_iterations=20000 --num_envs 2048 --run_name=reftrack2_s1
