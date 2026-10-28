#!/bin/bash
# Play ARM B -- omnidirectional WITH the commanded reference in the observation
# (Oct14_00-41-01_omniref_s1).
#
#   ./play_omniref.sh                # forward 0.6 m at the scan's best checkpoint
#   DX=0.4 ./play_omniref.sh         # the other command B actually hits
#   DX=0 DY=0.5 ./play_omniref.sh    # SIDEWAYS (B is omnidirectional; the scan only tested forward)
#   DX=-0.5 ./play_omniref.sh        # BACKWARD
#   CKPT=18000 ./play_omniref.sh     # the other checkpoint pulled locally
#   STAND=1 ./play_omniref.sh        # pure-stand test
#
# WHY CKPT 19500 AND DX 0.6
#   eval_ckpt_scan, 256 env, deterministic, pure torque, 5 checkpoints x 4 commands:
#     ck18000  0.4:0.10  0.6:0.88  0.8:0.00  1.0:0.00   netflight 0.580
#     ck18500  0.4:0.96  0.6:0.09  0.8:0.00  1.0:0.00   netflight 0.520
#     ck19000  0.4:0.51  0.6:0.24  0.8:0.00  1.0:0.00   netflight 0.510
#     ck19500  0.4:0.90  0.6:0.95  0.8:0.00  1.0:0.00   netflight 0.567  <== only ckpt over 0.9
#     ck20000  0.4:0.48  0.6:0.06  0.8:0.00  1.0:0.00   netflight 0.533
#   Note what that column does: 0.6 goes 0.88 -> 0.09 -> 0.24 -> 0.95 -> 0.06 across 500-iteration
#   steps. ARM A's 11 checkpoints were point-for-point identical by comparison. So B is UNSTABLE
#   checkpoint to checkpoint, and 20000 is one of its bad ones -- do not play the last checkpoint.
#   0.8 and 1.0 are zero at EVERY checkpoint: B's net flight tops out ~0.57 m, same wall as A.
#
# THE REFERENCE IS NOT OPTIONAL HERE
#   REF_PREVIEW_OBS=1 is what makes the observation 1060 wide; without it the env builds 1024 and the
#   checkpoint will not load. REF_SLOT_RADIUS=1 indexes the 61-bin table by the command RADIUS instead
#   of its x component -- that is the fix that took the clamp rate from 70.85% to 0.78% under a polar
#   command, and it changes which reference frame the policy is shown, so it must match training.
#   It reads SATA/paper_tools/commanded_ref_14700.npz; keep that file where it is.
#
# SAME THREE READING TRAPS AS play_omnisstill.sh
#   1. "replay pd_alpha=0.100" printed below is WRONG and ignores PD_OFF_PHASE. Real alpha is 0.
#   2. Only the FIRST jump uses your command; the auto-reset redraws it.
#   3. One jump is one sample -- and on THIS run adjacent checkpoints disagree wildly (see above).
set -e
cd /home/qiwang/torque_jump2/SATA/legged_gym
export PATH=/home/qiwang/miniconda3/envs/sata-clean/bin:$PATH
export LD_LIBRARY_PATH=/home/qiwang/miniconda3/envs/sata-clean/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1

export PD_TARGET_OBS=1          # part of the 1060-wide observation
export REF_PREVIEW_OBS=1        # the reference preview itself -- omit it and the load fails
export REF_SLOT_RADIUS=1        # index the table by radius, as trained
export DX_MAX=0.9               # as trained (B's command band, kept for faithfulness)
export GROWTH_WARMUP=96000 GROWTH_X0=192000   # replay-step schedule must match training
export PD_OFF_PHASE=all         # pure torque

RUN=Oct14_00-41-01_omniref_s1
CKPT=${CKPT:-19500}
export PLAY_LANDING_DX=${DX:-0.6}
export PLAY_LANDING_DY=${DY:-0.0}
export PLAY_LANDING_HEIGHT=${HEIGHT:-0.5}
export PLAY_LANDING_ADDED_MASS=${MASS:-0.25}
export PLAY_LANDING_STAND_ONLY=${STAND:-0}

echo "[play B] run=$RUN ckpt=$CKPT  cmd=($PLAY_LANDING_DX, $PLAY_LANDING_DY) height=$PLAY_LANDING_HEIGHT mass=+$PLAY_LANDING_ADDED_MASS kg"
echo "[play B] reference ON (REF_PREVIEW_OBS=1 REF_SLOT_RADIUS=1) -- expect a 1060-wide actor"
echo "[play B] pure torque via PD_OFF_PHASE=all -- the 'replay pd_alpha' line below is stale, ignore it"
exec python legged_gym/scripts/play_landing.py --task=go2_omnijump_landing_torque \
  --load_run=$RUN --checkpoint=$CKPT
