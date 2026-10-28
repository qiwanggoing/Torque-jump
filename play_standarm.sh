#!/bin/bash
# Play the STAND-ARM gate run (Oct09_04-24-48_standarm_s1) with a viewer.
#
#   ./play_standarm.sh            # jump at the default command
#   CKPT=18900 ./play_standarm.sh # a different checkpoint
#   DX=0.7 ./play_standarm.sh     # a different forward command
#   STAND=1 ./play_standarm.sh    # PURE-STAND test: cmd4=0.45 (<=0.5 -> NEVER jumps).
#                                 # This is the half the run was supposed to buy back
#                                 # (stand_no_takeoff: 5 airborne steps/episode -> 0.36).
#
# FOUR settings that each silently change what you are looking at:
#   REF_PREVIEW_OBS=1  this run descends from refprev2 (reference trajectory in the observation).
#                      Without it the env builds 1012 observations, the ckpt wants 1060, load fails.
#   PD_OFF_PHASE=all   play_landing replays the scaffold from the checkpoint's step_count, which
#                      rebuilds the step-count level but NOT _load_pd_scale. With the trained
#                      PD_FLOOR=0.2 that hands a pure-torque policy alpha_PD=0.100 back ->
#                      measured hit 0.00, launch angle +12 deg. Read as "the policy broke".
#   STAND_ARM_H0=0.30  _stand_arm_h is re-initialised in every new process and the latch cannot
#                      open inside a short play, so without this you are watching a GATED policy
#                      run with NO gate -- it will arm (and jump) from a lower stance than it
#                      ever did in training. STAND_ARM_H (the target) stays unset on purpose:
#                      the advance branch does min(target, nxt), so target=0 could zero the gate.
#   ADDED_MASS=0.25    NOT 0.0. This run has a dead point at EXACTLY nominal mass -- measured at
#                      ck17500, DX 0.7: nominal-0.25 kg scores hit 1.00, nominal+0.25 kg scores
#                      1.00, exact nominal records ZERO jumps (the robot just stands there).
#                      memory/project_ma_paper already lists this as the "精确标称质量死区" trap,
#                      and refprev2 does NOT have it at ck17600, so it is run-specific. The whole
#                      mass map is spiky (-1.0:0.25  -0.5:0.08  -0.25:1.00  0:none  +0.25:1.00
#                      +0.5:0.73  +1.0:1.00), so do not read one mass point as the policy's
#                      quality. Set ADDED_MASS=0.0 only to SEE the dead zone for yourself.
set -e
cd /home/qiwang/torque_jump2/SATA/legged_gym
export PATH=/home/qiwang/miniconda3/envs/sata-clean/bin:$PATH
export LD_LIBRARY_PATH=/home/qiwang/miniconda3/envs/sata-clean/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1

# ---- sstill6 env block, verbatim from run_standarm_s1.sh (programmatically diffed) ----
export GROWTH_WARMUP=96000 GROWTH_X0=192000 PD_FLOOR=0.2
export DX_MIN=0.3 DX_FLOOR_CURR=0 STAND_ANCHOR=1 STANCE_CURR=0
export PEAK_W=20 SUCC_LAND_SIGMA=0.15 SUCCDIST_CAP=1
export LOAD_FADE=1 PD_FADE_SCOPE=all LOAD_FADE_TIMED=1
export LOAD_FADE_STEP=0.1 LOAD_FADE_HOLD=12000 LOAD_FADE_GUARD=0.6
export PD_TARGET_OBS=1
export STAND_POSE_W=3.0
export STAND_STILL_W=6.0
export REF_PREVIEW_OBS=1
# ---- play-only ----
export PD_OFF_PHASE=all
export STAND_ARM_H0=0.30

RUN=Oct09_04-24-48_standarm_s1
CKPT=${CKPT:-18200}
export PLAY_LANDING_DX=${DX:-0.7}
export PLAY_LANDING_HEIGHT=${HEIGHT:-0.5}
export PLAY_LANDING_ADDED_MASS=${ADDED_MASS:-0.25}
export PLAY_LANDING_STAND_ONLY=${STAND:-0}

echo "[play] run=$RUN ckpt=$CKPT  DX=$PLAY_LANDING_DX height=$PLAY_LANDING_HEIGHT"
echo "[play] gate STAND_ARM_H0=$STAND_ARM_H0  PD_OFF_PHASE=$PD_OFF_PHASE  mass=+$PLAY_LANDING_ADDED_MASS kg"
exec python legged_gym/scripts/play_landing.py --task=go2_omnijump_landing_torque \
  --load_run=$RUN --checkpoint=$CKPT
