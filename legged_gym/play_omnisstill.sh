#!/bin/bash
# Play ARM A -- the omnidirectional run on the sstill6 baseline (Oct13_21-04-48_omnisstill_s1).
#
#   ./play_omnisstill.sh              # forward 0.5 m  (0.6 is OFF A CLIFF -- see below)
#   DX=0.4 ./play_omnisstill.sh       # a different forward command
#   DX=0 DY=0.5 ./play_omnisstill.sh  # SIDEWAYS. This run is omnidirectional; this is the new thing.
#   DX=-0.5 ./play_omnisstill.sh      # BACKWARD
#   CKPT=19200 ./play_omnisstill.sh   # a different checkpoint
#   STAND=1 ./play_omnisstill.sh      # pure-stand test (cmd4=0.45 -> never jumps)
#
# WHY DX DEFAULTS TO 0.5, NOT 0.6
#   eval_ckpt_scan (256 env, deterministic, pure torque) on ck18000/18200 gives a CLIFF, not a ramp:
#     hit[0.3:1.00  0.4:1.00  0.5:1.00  0.6:0.00  0.7:0.00  0.8:0.00  0.9:0.00  1.0:0.00]
#   sstill6, same harness, held 92-100% over 0.3-0.8. So the omnidirectional command cost ~0.3 m of
#   FORWARD reach, abruptly. Asking for 0.6 shows you a dog that cannot reach, not a broken policy.
#
# THREE THINGS THAT WOULD MAKE YOU MISREAD WHAT YOU SEE
#   1. The script prints "replay pd_alpha=0.100". THAT NUMBER IS WRONG -- it is computed from cfg's
#      growth fade only and knows nothing about PD_OFF_PHASE. go2_omnijump_curriculum_torque.py:352-366
#      makes PD_OFF_PHASE=all set _mult = 1 - ones = 0, and since the 2026-09-30 bugfix that per-env
#      weight really does reach the torque. The controller IS pure torque here.
#   2. Only the FIRST jump uses your command. resampling_time=20 holds it for the episode, but the
#      auto-reset after it calls _resample_commands and redraws distance (and, with OMNI_DEG, heading).
#      Ctrl-C and restart to see the same command again.
#   3. One jump is one sample. Adjacent checkpoints of this very run family have swung hit 0.43<->0.81.
#      Use this to look at the MOTION; use the scan for numbers.
#
# NOT SET ON PURPOSE (and what breaks if you add them)
#   REF_PREVIEW_OBS  A carries no reference. Setting it builds 1060 observations; the ckpt wants 1024.
#   STAND_ARM_H0     A never had the Oct09 stance gate (its run script says "MINUS STAND_ARM_H0").
#   OMNI_DEG         play pins the command by hand, so the polar wrapper would only randomise the
#                    heading on the SECOND episode. Leave it off and set the direction via DX/DY.
set -e
cd /home/qiwang/torque_jump2/SATA/legged_gym
export PATH=/home/qiwang/miniconda3/envs/sata-clean/bin:$PATH
export LD_LIBRARY_PATH=/home/qiwang/miniconda3/envs/sata-clean/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1

# Only these two matter for WHAT YOU SEE. Everything else in run_omnisstill.sh is a reward weight or
# a curriculum knob: play logs rewards without acting on them, and pins the command itself.
export PD_TARGET_OBS=1          # observation count -> 1024 actor / 399 critic. Omit it and the load fails.
export GROWTH_WARMUP=96000 GROWTH_X0=192000   # the replay-step计算 must match training's fade schedule
export PD_OFF_PHASE=all         # pure torque (see note 1)

RUN=Oct13_21-04-48_omnisstill_s1
CKPT=${CKPT:-18200}   # scan winner: longest net flight 0.555 m (19400 is only 0.421)
export PLAY_LANDING_DX=${DX:-0.5}
export PLAY_LANDING_DY=${DY:-0.0}
export PLAY_LANDING_HEIGHT=${HEIGHT:-0.5}
export PLAY_LANDING_ADDED_MASS=${MASS:-0.25}   # NOT 0.0: dodges the run-specific exact-nominal dead zone
export PLAY_LANDING_STAND_ONLY=${STAND:-0}

echo "[play] run=$RUN ckpt=$CKPT  cmd=($PLAY_LANDING_DX, $PLAY_LANDING_DY) height=$PLAY_LANDING_HEIGHT mass=+$PLAY_LANDING_ADDED_MASS kg"
echo "[play] pure torque via PD_OFF_PHASE=$PD_OFF_PHASE -- the 'replay pd_alpha' line below is stale, ignore it"
exec python legged_gym/scripts/play_landing.py --task=go2_omnijump_landing_torque \
  --load_run=$RUN --checkpoint=$CKPT
