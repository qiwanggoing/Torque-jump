"""Landing-point tracking built on top of the PROVEN vertical-jump env.

This task extends ``go2_omnijump_curriculum_torque`` (the validated SATA
vertical jumper, run May28_22-34-51) into a landing-point-conditioned jump.
The design principle is *additive*: the entire jump-driving reward stack of the
parent is inherited UNCHANGED — we only graft a thin landing layer on top.

Landing layer
-------------
- ``commands[0:2]`` are REPURPOSED from target horizontal VELOCITY (m/s) to
  desired landing DISPLACEMENT (meters) in the spawn / heading frame:
    commands[0] = dx  (forward landing distance)
    commands[1] = dy  (lateral landing offset)
  The robot spawns facing +x (identity heading), so the displacement maps
  directly onto world xy. The env sets ``landing_target = spawn_xy + (dx, dy)``
  every reset.
- The observation slot that used to carry the velocity command is replaced by
  the yaw-frame landing-point error  Ryaw^T (p* - p_base) = [fwd_err, lat_err, 0]
  (Olsen 2025 navigation signal). The 69-dim layout and the obs mirror-symmetry
  parity (err_x even / err_y odd, same as the old vx/vy/yaw command) are kept,
  so PPO sym_loss needs no change.
- Two new rewards:
    * ``projected_landing``  — dense, in-flight ballistic projection of the
      landing xy (Olsen densification; horizontal analogue of ``projected_peak``).
    * ``landing_position``   — sparse, terminal exp(-||land_xy - p*||^2 / sigma),
      gated by a real-jump peak so a legs-tucked fake jump cannot farm it.
- ``tracking_linear_velocity`` disabled: its target ``commands[0:2]`` is now
  meters, not m/s, so the velocity-tracking semantics no longer apply.

Curriculum
----------
- Stage 1 (``landing_stage = 1``): displacement ranges = [0, 0] → target == spawn
  → behaviour is the proven vertical jump that *also* learns to land in place.
  Used to confirm the graft did not break jumping.
- Stage 2 (``landing_stage = 2``): open the forward/lateral displacement ranges
  → the same rewards now drive directed (forward / diagonal) jumps.
"""

from legged_gym.envs.go2.go2_omnijump_curriculum_torque.go2_omnijump_curriculum_torque_config import (
    GO2OmniJumpCurriculumTorqueCfg,
    GO2OmniJumpCurriculumTorqueCfgPPO,
)


class GO2OmniJumpLandingTorqueCfg(GO2OmniJumpCurriculumTorqueCfg):
    # NOTE: the deeper 0.25 ready-crouch experiment was REVERTED — it triggered a collapse
    # (~iter 700, still in warmup/full-PD): the near-folded crouch was hard to hold, the policy
    # splayed the hips to balance, default_hip_pos collapsed, and the jump fell apart. Back to the
    # proven ~0.30 default stance (inherited). Revisit launch depth later via a milder crouch +
    # stronger default_hip_pos if pursuing more height.

    class env(GO2OmniJumpCurriculumTorqueCfg.env):
        # ATANASSOV-STYLE OBSERVATION HISTORY -- restored VERBATIM from commit 9ef99da (2026-08-18).
        # This is the ONLY from-scratch history configuration that has ever discovered the jump here
        # (run Jul31_17-37-21: squat_qualified 0 -> 0.96 at iter 908, mean_reward 30.7 vs flat's ~20).
        # Three earlier history attempts collapsed; the difference was NOT the actor stack but the
        # CRITIC -- see c_frame_stack below. Deliberately unmodified: its whole value is that it is the
        # verified point, so history_length / the stacked content / fatigue stay exactly as they were.
        #
        #   obs_buf = [ stacked_frame(49) x history_length | single extras(32) ]
        #
        # STACKED (x20 = 0.10 s at the 200 Hz pure-torque endpoint): kinematic state + PREVIOUS ACTION
        #   lin_vel3 + ang_vel3 + grav3 + dofpos12 + dofvel12 + foot4 + prev_action12 = 49
        #   -> lets the policy reason about its own dynamics (Atanassov's stated purpose).
        # SINGLE (current frame only, NOT stacked): command(landing_err3 + cmd_h1 + cmd4_1 + height2)
        #   + torques12 + motor_fatigue12 + pd_prior1 = 32. Atanassov keeps the command single (no 20x
        #   redundancy); we additionally keep the self-generated raw torques single, because stacking
        #   240 dims of them was one of the suspects when the first flatten-everything version died.
        num_stacked_frame = 49
        num_single_extras = 32
        history_length = 20
        num_observations = history_length * num_stacked_frame + num_single_extras   # 49*20 + 32 = 1012
        # ⭐ ASYMMETRIC CRITIC -- this is the change that actually fixed history here. A critic fed the
        # FULL actor stack overfits the early low-return rollouts (value_loss -> 0.002, the signature),
        # which wrecks the advantages and stops the discovered squat from ever being reinforced. Feed it
        # a SHORT stack of PRIVILEGED frames instead (same c_frame_stack=3 as the working my_go2_jump).
        #   priv frame(121) = stacked_frame(49) + extras(32) + priv_extra(40)
        #   priv_extra(40)  = root_z1 + base_lin_vel3 + feet_pos_local12 + feet_vel12 + feet_forces12
        c_frame_stack = 3
        single_num_privileged_obs = num_stacked_frame + num_single_extras + 40      # 49+32+40 = 121
        num_privileged_obs = c_frame_stack * single_num_privileged_obs              # 3*121 = 363

    class control(GO2OmniJumpCurriculumTorqueCfg.control):
        # Step H: turn ON the dual-head aux-stabiliser torque path in _compute_torques.
        # (Default False in the parent -> all other tasks with num_actions=12 are untouched.)
        aux_stabilizer_head = True

    class asset(GO2OmniJumpCurriculumTorqueCfg.asset):
        # 2026-08-07: use the REAL Go2 joint ranges instead of the SATA URDF's truncated thigh
        # ([0,1.5]/[0,2.0] vs official [-1.5708,3.4907]/[-0.5236,4.5379]). model_4600 was measured
        # sitting on those fake stops for the whole jump (FL_thigh pinned at 1.500, RL_thigh at
        # -0.006), which is both a learned crutch and a major sim2sim gap -- MuJoCo/hardware have
        # the real, much wider range. See GO2Torque.OFFICIAL_DOF_POS_LIMITS / SOFT_DOF_POS_BAND.
        # REQUIRES A RETRAIN: old checkpoints were trained against the walls.
        official_dof_pos_limits = True

    class domain_rand(GO2OmniJumpCurriculumTorqueCfg.domain_rand):
        # RE-CENTER base-mass DR on the URDF-nominal robot (2026-07-05). The inherited range
        # [-1, +5] (mean +2kg) was HEAVY-BIASED: the mass-BLIND policy (mass not in obs) tunes ONE
        # push to the distribution's bulk (~+2kg), so it nails +2..+4kg (eval hit 1.00) but MISSES at
        # nominal (added=0 = the real URDF robot: eval hit 0.00, systematic undershoot). Proven with
        # eval_isolate massfix sweep. Real robot == URDF nominal, so center the range there. Keeps
        # randomize_base_mass=True (real payload/battery/tolerance margin), just symmetric about 0.
        # If the retrained policy is weak at the ±edges, TIGHTEN this or add base-mass to the obs
        # (proper fix for a mass-sensitive jump: let the policy ADAPT the push to the actual mass).
        added_mass_range = [-1.0, 1.0]

    class growth(GO2OmniJumpCurriculumTorqueCfg.growth):
        # PD fade EARLY (pure torque by ~iter500). The MIDDLE-window experiment (warmup 100000/x0 240000,
        # full PD until iter1450) FAILED: run Jun10_00-50-26 stayed squatQ=0 / peak 0.15 / NO jump for
        # 1087 iters even with full PD. Cause: with the squat-QUALIFIED gate, a 50% PD prior only gives a
        # shallow squat that never qualifies -> jump chain stays locked -> RL gets no jump-reward signal,
        # and the strong PD keeps it "comfortable" not squatting deep. Early fade WORKS precisely because
        # PD leaves fast and FORCES the RL to learn a qualifying squat-jump (Jun09_19-29-38: succ 0.90,
        # peak 0.57, continuous 0.65 jumps in play). The early-fade oscillation is cosmetic -- it still
        # converges to a great policy. ("PD-longer = smoother" held for Jun03 only because that older
        # config had no squat-qualified gate.) general_scale ramps 0->1 linearly warmup_steps->x0;
        # pd_alpha = 0.5*(1-general_scale). step_count ~= 69/iter.
        # SLOW GRADUAL FADE (user: "课程进化减慢从200到1200"): full PD only until ~iter200, then a long gradual
        # ramp to pure torque by ~iter1200, so the policy ADAPTS to pure torque smoothly instead of a sharp
        # transition (the iter~1100 dip was the PD/freq/torque transition). step rate ~96-97/iter (measured) ->
        # warmup 19200 ~= iter200, x0 115200 ~= iter1200 (verify the REAL general_scale trajectory from the run;
        # rate accelerates as freq ramps so it may complete a bit before 1200).
        # ⚠️ HISTORY: a MUCH slower PD-slow (warmup 100000 / x0 240000 = full PD until ~iter1450) FAILED HARD --
        # squatQ=0, NO jump for 1087 iters, because prolonged FULL PD gives a shallow squat that never qualifies
        # -> jump chain locked. THIS setting keeps the FULL-PD window SHORT (~iter200) to dodge that trap, then
        # fades slowly. WATCH the first ~iter300: if squatQ stays ~0 / no flight -> the short full-PD window is
        # still too long; shorten warmup. If it jumps fine and the iter~1100 dip is gentler -> the slow fade worked.
        # REVERTED to the ORIGINAL fast fade (user): PD should fade EARLY so the PURE-TORQUE policy is
        # established first, THEN the dx_max curriculum evolves ON pure torque (with the safety-revert in
        # _update_dx_curriculum so it plateaus instead of crashing, and re-advances as the pure-torque policy
        # gets stronger). The slow fade (warmup 19200/x0 115200) made dx_max hit the ceiling while PD was still
        # on (general_scale 0.45) -> the reach it found was PD-assisted, not the real pure-torque reach.
        # REVERTED to the Jun23_01-23-30 baseline (user): SLOW gradual fade -- full PD until ~iter200, then a
        # long ramp to pure torque by ~iter1200 (step ~96/iter -> warmup 19200 ~iter200, x0 115200 ~iter1200).
        warmup_steps = 19200       # full PD 到 iter200. ⚠️step/iter 非线性! step()内 `while current_dt·freq<1` 每env.step跑
                                   # 200/freq个substep(每个substep step_count+=1): warmup期 general_scale=0/freq=100 -> 2/env.step
                                   # = 96/iter -> warmup19200=iter200. fade末 freq=200 -> 1/env.step=48/iter. (旧注释"96/iter"只对warmup期.)
        x0 = 70400                 # 纯力矩 iter1000 (user). fade期 freq100->200 => step/iter 96->48,积分得【fade完成iter = 200 +
                                   # (x0-19200)/64】. 实测x0=48000->iter650✓,要iter1000 -> x0=70400. fade跨度iter200->1000(800iter).

    class commands(GO2OmniJumpCurriculumTorqueCfg.commands):
        # Landing-point task: commands[0:2] repurposed velocity -> landing displacement (m).
        # Stage 1 keeps the displacement at [0,0] (land in place == proven vertical jump).
        # Set landing_stage = 2 to open the ranges below; the env widens
        # command_ranges["lin_vel_x"/"lin_vel_y"] accordingly at init.
        landing_stage = 2                      # STAGE 2 ON: env widens lin_vel_x/y ranges to the disp ranges below.
        # Anchor for the landing target locked at takeoff. False = squat bottom (the anti-cheat anchor:
        # stops the creep from ADDING distance, but the target walks forward WITH the creep so creeping
        # stays free). True = the xy the jump was commanded from -> every metre crept is a metre of
        # landing error. Only meaningful at dx=0 (see the note in _update_jump_state) -> stage 1 sets it.
        landing_anchor_jump_start = False
        # 2026-08-12: [0.5, 1.5] -> [0.4, 1.2]. The old range asked for MORE THAN THE ROBOT CAN FLY, and
        # that is what forces the run-up: the landing target is anchored at the squat bottom, so hitting
        # a 1.2 m command when the clean flight tops out at 0.6-0.75 m REQUIRES shuffling the remaining
        # 0.4 m on the ground. Accuracy and a clean single push were literally asking for opposite
        # things, and no weight can settle that (anchoring at the jump command instead does not help
        # either -- creep 0.4 + fly 0.8 still lands on target). Measured at dx=1.2: target 0.95 m ahead,
        # flight 0.59 m, creep 0.40 m, 17 foot-contact flips before takeoff.
        # Upper bound kept at 1.2 (user) rather than dropped to the current ~0.9 capability, to keep
        # pressure to reach further; 0.75-1.2 therefore REMAINS a conflict zone where creeping is still
        # the only way to be accurate. Watch which command bin run_up_creep comes back in.
        # Also see [[project_curriculum_overshoot_collapse]]: commanding past the real reach drives
        # value_loss up and eventually gets punched over by the exploration-noise hump -- the window run
        # already sat at noise_std 0.159 vs 0.081, which is that signature.
        # ⚠️ 2026-08-18 REVERTED [0.4,1.2] -> [0.5,1.5] (user): the new baseline for the step-by-step
        # plan is "model_4600 + the joint-limit fix and NOTHING ELSE", so that every later step has ONE
        # variable. That base is already trained (run Aug08_16-35-49 @ 5bad5e4): clean_reach 0.484,
        # creep 0.190, reliable_reach_dx 0.646, squatQ 0.959, reward 17.1, noise 0.081. The command-range
        # change (and the payment window below) go back on the table AFTER the observation/reference work,
        # measured against that base rather than bundled with it.
        landing_disp_x_stage2 = [0.5, 1.5]    # FIXED forward range (2026-07-11, user): no curriculum-from-0 -> command
                                              # 0.5-1.5 m directly from the start. Goal = FARTHER: every command is far, so
                                              # forward_reach (capped-at-command) always pays for jumping as far as possible,
                                              # pushing toward the physical reach instead of the conservative curriculum's
                                              # parked ~0.6 m. ⚠️ discovery: the jump is still bootstrapped by squat/launch/
                                              # height rewards (which don't need the landing point), but if early flight_rate
                                              # stays 0, lower the floor (e.g. 0.2) or add a brief warmup.
        landing_disp_y_stage2 = [-0.30, 0.30]  # IN-PLANE OMNIDIRECTIONAL (2026-07-11): forward + side + diagonal. d_y is a
                                               # fixed uniform range (no curriculum); d_x keeps its curriculum. Diagonal = both
                                               # non-zero. Paired with a command-conditional default_hip_pos (relax the hip
                                               # lock for lateral commands so the hips can abduct to push sideways). Yaw NOT
                                               # unlocked yet (ang_vel_yaw=[0,0]); yaw-turning needs its own new rewards.

        # ---- DISTANCE CURRICULUM (Atanassov 2025 local-difficulty) ----
        # Start the forward dx range at 0 (pure in-place = the proven vertical-jump discovery; the
        # landing reward is fully available because target==spawn) and grow the upper bound one
        # `step` at a time. Advance ONLY when, at the current distance, the policy both lands safely
        # (successful_jump_rate) AND lands near the commanded point (landing_hit_rate, |land-target|
        # <= hit_tol) — the hit gate stops the curriculum from outrunning the policy (success alone
        # is height-only and would let an in-place policy keep advancing). After each bump both rates
        # dip and must be re-earned at the new distance. Trains forward jumping in ONE from-scratch
        # run without the discovery cliff that a one-shot dx[0,0.40] open hits.
        landing_dx_curriculum = False        # OFF (2026-07-11, user): no curriculum-from-0. Command is drawn uniformly
                                             # from landing_disp_x_stage2 = [0.5, 1.5] m (see else-branch in _init_buffers).
        # BIASED command sampling (Atanassov local-difficulty): concentrate most jump commands at the FAR
        # frontier (the goal = farthest landing point) instead of uniform over [0, dx_max]. The policy then
        # practices mostly where it counts; a spread fraction is kept for the easy->hard gradient + retention.
        # Only DISTANCE is biased (height untouched -- goal is a STABLE landing at the farthest point).
        landing_dx_biased = False              # DISABLED: 70/30 biased BACKFIRED (run iter4803) -- hit crashed 0.64->0.12,
                                               # dx_max regressed 1.1->0.9, stable_cum ~0. Concentrating on far STARVED the
                                               # near/mid commands that bootstrap distance-conditioning, and at far an UNDERSHOOT
                                               # makes the distance-normalized accuracy reward VANISH (exp(-big)~0) -> no gradient
                                               # -> the policy never learned to modulate distance (fixed high jump, lands off-target).
                                               # Uniform provides the easy->hard learning ladder -- keep it. The far-accuracy ceiling
                                               # (~1.1, hit 0.64) is a REWARD-gradient problem, not a sampling one.
        landing_dx_frontier_frac = 0.7         # (inert while landing_dx_biased=False)
        landing_dx_frontier_lo = 0.8           # (inert while landing_dx_biased=False)
        landing_dx_start = 0.0                 # initial dx upper bound (0 = in-place)
        landing_dx_final = 2.0                # final dx upper bound (the Stage-2 target)
        landing_dx_step = 0.10                 # [global advance-only — 被 per-env 双向课程取代, 见下]
        # ── PER-ENV 双向课程 (2026-07-04, user, Atanassov/terrain-curriculum 风格) ──
        # 根治 dx 虚高: 不再全局单值+只升(会被PD辅助期+noise冲高、advance-only不退). 改成每个 env 一个自己的
        # 上界 landing_dx_env, 命令从 [0, 自己上界] 抽; 落地后只看"挑战命令"(dx>=per_env_far_frac×自己上界):
        # 命中→上界+step_up, 脱靶→−step_down. 升慢降快 → 数学上收敛到"挑战命中率≈step_down/(step_up+step_down)"
        # 的距离 = 该 env 真能稳命中的上界. PD辅助/noise冲上去的, 纯力矩后跳不到→自动降级收敛回真实~0.6, 无需门/gate.
        landing_dx_percurr = True              # 开 per-env 双向课程(取代 global advance-only)
        landing_dx_step_up = 0.02              # 命中挑战命令 → 自己上界 +这么多
        landing_dx_step_down = 0.18            # 0.14→0.18 (2026-07-11, 抬自限门到90%): 升:降=1:9 → 收敛到挑战命中≈90%
                                               # (=用户"门槛调到0.9"). 平衡命中率 = step_down/(step_up+step_down) = 0.18/0.20 = 0.90.
                                               # 更保守 → 每个env上界停在"90%可靠"的距离、离够不到的边缘更远 → 命令更少落进够不到区 →
                                               # 更少毒化策略 → 治后期崩(overshoot→毒化→塌). [0.10→0.14 史: 升档贴近确定性能力].
        landing_dx_per_env_far_frac = 0.6      # 只 dx>=0.6×自己上界的"挑战命令"结果决定升降(近端命令不影响,防虚升)
        landing_dx_floor = 0.0                 # 上界下限(不降到负)
        # ── FRONTIER PROBE (2026-07-04, option-1, 配 forward_reach 2×): 一小撮 env 命令探到自己上界之外
        # [dx_env, dx_env×probe_hi], 让 forward_reach(往命令方向够更远)+takeoff_velocity_match(往命令v_req更狠launch)
        # 的"跳更远"梯度在前沿变活(命令不超上界时这俩休眠). 探测命令 EXPECTED 够不到, 排除出课程升降(不污染诚实 dx_env).
        landing_dx_probe_frac = 0.0            # 撤回 option-1: 0.25→0 (probe 关). 保留参数, 以后 RSI 阶段可能再用.
        landing_dx_probe_hi = 1.4              # 探到 dx_env×1.4 (probe_frac=0 时 inert)
        # COMBINED advance gate: advance only when the SAME jump both lands on target AND lands
        # stably (landing_stable_hit_rate). Replaces the old two separate thresholds (succ + hit),
        # which let "hit-then-topple + short-but-stable" pass without any jump being both -> the
        # curriculum blew through to the cap. (succ/hit thresholds below are now unused.)
        landing_dx_stable_hit_threshold = 0.80 # advance needs CUMULATIVE far-band stable-hit rate >= this. 0.70→0.80
                                               # (user, treat dx虚高): 0.70太松→far-band命中被noise偶冲过就升→advance-only堆到1.2虚高
                                               # (真实力矩只squat→land~0.76). 0.80=要far-band真稳定命中才升→dx_max自停在~0.85-0.9够得到处.
        landing_dx_min_far_samples = 400       # 150→400 (user): 样本太少(150)→cum_rate被小样本noise冲过门. 400=大窗口压noise. so the
                                               # gate reflects SUSTAINED mastery, not a noisy few-sample spike
                                               # (the old per-batch EMA spiked to thr on 1-2 jumps -> over-advanced
                                               # dx_max to 1.6 with only ~0.5 real far-band rate -> late collapse).
        # FAR-BAND: the stable-hit rate is measured ONLY over jumps whose commanded dx fell in the
        # top fraction [dx_max*(1-far_frac), dx_max] of the open range -> the gate requires the
        # NEWEST/farthest distances to be stably hit, not the easy near commands carrying a uniform
        # average. (uniform averaging let dx_max reach 1.2 while really mastering ~0.9.)
        landing_dx_far_frac = 0.30             # 0.20 -> 0.30: WIDEN the far band back a notch. 0.20 was TOO narrow:
                                               # ~1 far-band sample per log window (fb_n~1) -> the advance metric got
                                               # NOISY -> a lucky streak over-advanced dx_max from 0.9 to 1.0, past the
                                               # reliable reach (fb_hit cliff 0.79@0.9 -> 0.15@1.0 -> late decline).
                                               # 0.30 keeps "hit near the edge" but with enough samples for a STABLE
                                               # gate, so dx_max settles at the reliable reach instead of overshooting.
        landing_dx_succ_threshold = 0.80       # [unused — superseded by landing_dx_stable_hit_threshold]
        landing_dx_hit_threshold = 0.55        # [unused — superseded by landing_dx_stable_hit_threshold]
        landing_dx_hit_tol = 0.10              # 0.07→0.10 (2026-07-05, HONEST 校准): 与确定性 eval 的命中定义(err<=0.10)对齐.
                                               # 铁证(eval_reach_ceiling, model_3000): tol=0.10 下 cmd0.6 确定性命中 0.88 → 真实能力=0.6.
                                               # 而 0.07 过紧, 把 dx_env 压到真实能力(0.6)以下(实测 dx_mean 收敛 0.48 而非 line130 设计意图的 ~0.6).
                                               # 0.07 的初衷"防松松够到虚高"是对的, 但确定性 eval 证明 10cm 才是真实操作容差, 非虚高. 只动这一个,
                                               # step_down 留 0.14 (若 retrain 后 dx_mean 仍 <0.55 再松 step_down). per-env 下 step_up=0.02, 原
                                               # "tol<step" 不变式(为 global step=0.10 写)不再约束: in-place 落点(~0.15)距任何远命令 err>>tol, 不会假命中.
        landing_dx_ema_alpha = 0.02            # EMA smoothing on the per-reset-batch stable-hit rate
        landing_dx_min_hold_steps = 1500       # min policy-steps held at a stage before it may advance (~30 iters)
        # Per-resample STAND probability: each resample (every resampling_time=1.8s) the robot STANDS
        # if commands[4] <= jump_command_threshold (0.5). Default range [0,1] -> 50% stand. Narrow to
        # [0.45,1.0] -> ~9% stand, so the robot idles far less and jumps almost every resample. (The
        # IMPORTANT standing — recovering to a stable stand after landing — is still trained in every
        # jump episode's post-landing buffer.)
        jump_command_range = [1.0, 1.0]    # 单跳只练跳(user): cmd4恒1、无站立episode (去掉cmd=0训练). cmd4 二值 via stand/jump_command_value.
        # BINARY jump command: 1.0 = jump, 0.0 = stand. The old scheme put STAND at the sampled [0.45,0.5]
        # band -- right under the 0.5 threshold -- and at a near-threshold stand command (e.g. 0.49) the
        # policy HESITATES and twitches a foot off (the stand-episode in-place hop). Pinning stand->0 and
        # jump->1 makes cmd4 an unambiguous binary far from the threshold, and drops the meaningless (0.5,1.0]
        # jitter from the cmd4 obs feature (its magnitude is never used as effort -- only cmd4>threshold).
        # Verified: at cmd4=0.45 the trained policy stands 800 steps dead still; 0.49 twitches. Needs RETRAIN
        # (the current model never saw a clean 0/1 -> OOD in play).
        stand_command_value = 0.0
        jump_command_value = 1.0
        # The moment the robot touches down, flip the jump command to STAND for the whole 0.75s landing
        # buffer (instead of holding cmd4=1 until the jump "finishes"). Without this the policy sees cmd4=1 +
        # the residual landing error during the buffer and HOPS to chase the undershot target. Verified in
        # play (force cmd4=0 at touchdown) that this kills the post-landing chase-hop. Landing-accuracy
        # rewards key off self.landing / touchdown-locked landing_root_xy, not cmd4, so scoring is unaffected.
        disable_jump_on_landing = False    # 单跳只练跳(user): 落地不切cmd4=0 (去站立). ⚠️去掉了 post-landing chase-hop 防护, 落地欠程时可能hop追目标, 观察.
        single_jump_command_prob = 1.0     # 单跳: 一个episode一跳、跳完停站立 (撤回连续跳的 0.0). single-jump(Jun06) 比
                                           # continuous(Jun09 noisy bistable succ/flght 0.28-0.56) 干净(flght0.97-1.0/succ0.77-0.86平滑);
                                           # 连续跳落地立刻再跳没法蓄力 -> 每跳只 0.13m; 单跳能蓄力 -> 纯力矩~0.77m.

        class ranges(GO2OmniJumpCurriculumTorqueCfg.commands.ranges):
            jump_height = [0.40, 0.50]   # 0.60 -> 0.50 (2026-07-10, RE-APPLY the LAUNCH-ANGLE fix): 0.60 launched at
            lin_vel_x = [0.0, 0.0]       # repurposed: landing dx (m). Stage 1 = land in place.
            lin_vel_y = [0.0, 0.0]       # repurposed: landing dy (m). Stage 1 = land in place.
            ang_vel_yaw = [0.0, 0.0]

    class rewards(GO2OmniJumpCurriculumTorqueCfg.rewards):
        # Landing-reward kernel widths + real-jump gate for the sparse terminal term.
        sigma_pos_landing = 0.06            # Stage-2: TIGHTENED from 0.12. At 0.12 an in-place jump at cmd dx=0.40
                                            # (err=0.16) still earned exp(-1.33)=0.26, and at the avg cmd dx=0.20
                                            # (err=0.04) earned 0.72 -> in-place farmed ~70% of the landing bonus.
                                            # 0.06 cuts those to 0.07 / 0.51 -> forces real forward motion to score.
        sigma_landing_proj = 0.05           # Stage-2: TIGHTENED from 0.10 (same reason). In-place at cmd dx=0.40
                                            # drops exp(-1.6)=0.20 -> exp(-3.2)=0.04; gradient at in-place still alive.
        # DISTANCE-NORMALIZE the landing kernels (Yang 2023): err is divided by the commanded
        # displacement^2 so the reward is SCALE-INVARIANT. Fixes the ~constant RELATIVE undershoot
        # (lands at ~85% of cmd) -- a fixed-sigma kernel's gradient vanishes at far targets, so the
        # policy plateaus short; normalized, a 15% miss at 1.2m is pushed as hard as at 0.4m.
        landing_err_normalize = True
        sigma_landing_proj_norm = 0.025     # kernel width on RELATIVE squared err (proj). ~5% miss->0.90, 15%->0.41
        sigma_pos_landing_norm = 0.04       # kernel width on RELATIVE squared err (terminal landing_position)
        landing_norm_dist_floor = 0.30      # min distance used in the normalizer (in-place/near cmds judged vs 0.30m)
        # landing_stability BRAKE kernel widths (read by _reward_landing_stability). MOVED here from class scales
        # where they were dead (cfg.rewards.<name> missing -> default 0.25 -> brake reward ~0 at landing speeds).
        # Loosened so the brake has gradient: |v|~2.5 -> exp(-2.5^2/2.0)~0.04 rising to 1.0 as it stops on the spot.
        landing_stability_lin_sigma = 2.0   # was effectively 0.25 (floored to ~0 for real landing |v|, no gradient)
        landing_stability_ang_sigma = 1.0   # was effectively 0.5
        four_leg_push_force_floor = 200.0   # N: total vertical GRF below which four_leg_push is NOT graded -> only the
                                            # REAL push (> body weight ~147N), not the static squat-load where legs
                                            # merely bear weight (which would be farmable by lingering in the squat).
        four_leg_push_force_target = 90.0   # N: per-leg vertical GRF for FULL credit (saturates -> surplus free, no
                                            # front==rear constraint). ~2.4x the static per-leg share; a leg below this
                                            # while still on the ground = "idling" -> graded down -> the policy wakes it.
        # NON-VANISHING far PULL on projected_landing (see _reward_projected_landing): the exp kernel ->0 for
        # a big undershoot at a far target -> no gradient -> curriculum plateaus (~1.3). A linear term gives
        # partial credit + a constant slope toward the target at any distance, so the policy keeps learning to
        # reach far. exp = precision near; linear = "reach to it" far. Discovery-safe (gated on a real jump).
        landing_lin_pull = True
        landing_lin_coef = 1.0              # 0.5 -> 1.0: STRENGTHEN the far pull. The far-band stalled at dx 1.0
                                            # (far accuracy ~0.47 < 0.70 gate); the policy undershoots far cmds to
                                            # ~0.85 because the reach-to-target pull was too soft vs the cost of a
                                            # bigger launch. Doubling coef doubles the far-undershoot slope
                                            # (coef/d_ref 0.33->0.67 /m) AND the far reward (cmd 1.0 land 0.6:
                                            # projected_landing 0.37->0.73). exp precision peak near target unchanged.
        landing_lin_ref = 1.5              # m: linear runs from 1 (on target) down to 0 at this miss distance
        # Takeoff-omega suppression (see _reward_base_ang_vel_xy): once succ_rate EMA >= gate, LATCH a stronger
        # ω penalty that also covers the PUSH -> kill the nose-down spin at the SOURCE (takeoff) so the body
        # flies level and lands flat. succ-rate gate (not a fixed step) = adapts to discovery speed, never
        # blocks the messy from-scratch pushes (an ungated strong ω penalty broke discovery, iter526 flight0).
        takeoff_omega_succ_gate = 0.80     # latch the stronger ω penalty once succ_rate EMA clears this
        takeoff_omega_gain = 4.0           # post-gate multiplier on base_ang_vel_xy (-0.15 -> ~-0.6 effective)
        # YAW into the quadratic omega penalty (2026-08-25). 1.0 = a yaw rate costs exactly what the same
        # roll rate costs ("rotation is rotation"); 0.0 = off, which is what every other config gets.
        # Rationale and measurements: see _reward_base_ang_vel_xy. Two properties make this the targeted
        # fix rather than the mirror-loss hammer: the gradient grows with omega instead of vanishing, and
        # it forbids only SPIN, not left-right asymmetry in general (which the sym ablation suggests is
        # part of what buys reach: flat sym-off tracked commands at dAir +0.21..0.31 vs +0.03 with sym on).
        # It also inherits the existing discovery-safe staging for free: before the succ-rate latch it acts
        # on flight+landing only, and after the latch `_takeoff_omega_on` extends it to the PUSH phase at
        # x4 -- which is where the angular momentum is actually created (yaw hits -91 deg BEFORE takeoff).
        # WATCH: this term is quadratic and a 7.8 rad/s spin is ~61 rad^2/s^2, so early on it can be large.
        # A clean jump pays ZERO (it penalises rate, not airtime), but if flight_rate / squat_qualified
        # collapse it means exploration is being taxed too hard -> drop this to ~0.25 before blaming
        # anything else.
        base_ang_vel_yaw_weight = 1.0
        # HARD pitch termination (see check_termination): end the episode if the base pitches NOSE-DOWN beyond
        # this (projected_gravity[:,0], ~sin(tilt)) AT TOUCHDOWN (the landing phase). Forces a level touchdown
        # (no front-feet-first), since soft penalties got traded off. Same succ-rate gate as takeoff_omega
        # (only after the robot can jump). 0 = off. MEASURED (play, model_10000): touchdown ~0.47 (28deg) nose-
        # down, consistently. 0.40 (~24deg) is a modest step below that -> forces ~4deg flatter (low collapse
        # risk); tighten over runs (0.40->0.35->...) to progressively flatten; loosen if dx/succ collapse.
        landing_tilt_terminate = 0.0       # OFF for now -- trying the SOFT (reward) route first (see below); flip to ~0.40 if soft fails
        # first_jump_delay_steps stays at the inherited 55 (0.275s). A 1s pre-jump idle (200) was
        # tried and BROKE from-scratch discovery (iter774 flight=0 vs the proven run's 0.914 by
        # iter500): 1s of standing rewards makes "don't jump" too comfortable -> the policy never
        # risks the squat-then-push (same failure mode as Jun09_11-29-05 strong default_pos/yaw).
        # The 1s settled-stance is a PLAY/visual nicety only -> set it in play_landing, not training.
        landing_real_jump_min_peak = 0.40   # peak gate for the landing_position reward
                                            # (omnijump squat settles ~0.31, real jump peaks ~0.56)
        landing_buffer_steps = 150          # was 25 (=0.125s, inherited). A jump only "finishes" (success
                                            # credited + next jump re-enabled) after the robot stays stable
                                            # 150 steps (~0.75s) post-touchdown without exceeding fallover tilt.
                                            # = the "stand stable" requirement. The 25-step buffer let the policy
                                            # get credit after 0.125s then topple ~0.75s later (play roll_cutoff).
                                            # Toppling within these 150 steps -> roll_cutoff termination = penalty.
        # CONTINUOUS-jump NEXT-JUMP pose gate (DECOUPLED from success): success/finish is granted on
        # the time buffer alone (dense discovery signal preserved); this gate only delays the NEXT
        # jump until the robot has RETURNED to the default standing pose. Forces every jump in a
        # continuous sequence to start from the same canonical idle stand (kills chain drift) WITHOUT
        # withholding the successful_jump carrot from the still-learning policy. (The earlier version
        # gated success itself on pose -> starved discovery -> never learned to jump; Jun09_11-29-05.)
        # Gate metric = sum|dof - default_dof_pos| over the 12 joints. Threshold is non-critical now
        # (too tight = fewer jumps/episode, NOT stuck; too loose = no-op).
        next_jump_requires_default_pose = False   # continuous-only gate; no-op in single-jump mode (no "next jump").
        next_jump_default_pose_threshold = 1.5
        # Heading hold: keep tracking_angular_velocity active even at zero yaw command (commands[2]=0) so
        # its error term = wz^2 = a yaw-rate damp during flight -> stops the heading drift. (OmniNet does
        # this; our base reward otherwise only rewards tracking a NONZERO commanded yaw.)
        ang_vel_damp_zero_command = True
        # AIRBORNE-only: only damp yaw while actually in the air, NOT during the squat-but-not-taken-off
        # phase. Otherwise the strong (1.5) yaw reward pays the policy to sit in the squat holding still
        # and never jump -> discovery collapse (Jun10_12-47-57).
        ang_vel_damp_airborne_only = True
        # FIX: tracking_linear_velocity_all_time inherits True from omnijump_torque, but in
        # _reward_tracking_angular_velocity that branch is checked FIRST and OVERRIDES ang_vel_damp_airborne_only
        # above -> the 1.5-weight yaw damp wrongly stayed active during squat + landing (not just the air).
        # Force False here so the airborne-only gate actually takes effect (yaw damp in the air only, as
        # intended). The linear-velocity reward that also reads this flag is weight 0, so this is a no-op there.
        tracking_linear_velocity_all_time = False
        projected_landing_min_height = 0.40 # instantaneous height gate for the DENSE projected_landing:
                                            # blocks the legs-tucked sprawl farm (body ~0.13, feet off ground)
                                            # while keeping dense in-place landing control during real apex.
        # pose_guidance_sigma for joint_angle_aerial/prelanding/landing: kept inherited 5.0
        # (sharpening to 2.0 backfired — sharp exp saturates at the large air-pose error; the
        # fix for weak pose rewards is WEIGHT 1.5, not sigma). NOTE: the old joint-based
        # pushoff_leg_sync was replaced by contact-based foot_contact_sync, which uses no sigma.
        pose_guidance_sigma = 5.0

        # ---- Countermovement via a STANCE-SQUAT shaping reward (Atanassov 2025) ----
        # Root cause of "stand-and-pop" (no dip, capped height): nothing rewarded dipping
        # BEFORE the push, so the policy sat in the local optimum Atanassov explicitly warns
        # about ("standing in place"). Atanassov breaks it with RSI + a dense "squat to 0.2m
        # while on the ground" reward; we only had RSI. Fix = add that squat reward.
        # CRITICAL: its gate has NO vz<=0 condition. The old joint_angle_loaded used
        # phase_loaded (jumping & ~taken_off & vz<=0) — a DEAD LOOP: it only fired once the
        # robot was already dipping, so nothing ever drove the dip. See _reward_stance_squat.
        stance_squat_sigma = 0.02           # exp kernel width on (base_z - stance_squat_height);
                                            # 0.02 gives a strong gradient from the 0.31 stand
                                            # down to the 0.20 squat (stand value ~0.55 -> 1.0).
        # Squat-POSE gate (GUIDE to the target, replaces the gameable base_z height gate):
        # "squatted" = whole-body joint pose within squat_pose_threshold (L1 over 12 joints) of the
        # loaded pose q_squat. Standing is ~7.1 rad from q_squat (calf -1.5->-2.66, thigh 0.8/1.0
        # ->1.53, hips unchanged); q_squat itself = 0. Drives stand->squat, then unlocks the jump.
        squat_pose_sigma = 5.0              # exp kernel on |dof - q_squat| for the dip reward. Raised 3->5: the
                                            # pull from standing (7.1 rad away) is (1/sigma)*e^(-7.1/sigma), which
                                            # is ~55% stronger at 5 than 3 (peaks near sigma~7). default_pos no
                                            # longer competes during the dip, so this positive pull now drives it.
        squat_pose_threshold = 3.2          # was 2.8: EASED (stuck @ squatQ~0.48). "in the squat" = pose_err<=3.2, shallower from
                                            # standing (7.1). THE depth knob: stuck-not-jumping (can't fold
                                            # enough) -> RAISE; jumps too shallow / want a deeper load -> LOWER.
        default_hip_pos_lat_ref = 0.15      # read by _reward_default_hip_pos override (lateral unlock). The |d_y| (m) at which
                                            # the hip-abduction lock is fully released; forward (d_y=0) keeps the full lock.
                                            # Lives here in `class rewards` (NOT in scales, or it'd be mis-read as a reward term).
        squat_hold_steps = 25               # was 40 (0.2s) -> 25 (0.125s): EASED to unstick. jump chain unlocks only after the squat POSE is HELD within
                                            # squat_pose_threshold for this many CONSECUTIVE steps (= 0.2s at
                                            # sim dt 0.005s). Closes the "flick through the pose for one frame
                                            # and harvest the flight" hole. THE dwell knob: collapses to
                                            # not-jumping (can't hold long enough) -> LOWER (e.g. 20=0.1s);
                                            # want a more deliberate load before launch -> RAISE.
        # Give the dip+push room: a countermovement (~0.3-0.35s) does not fit the old 40-step
        # (0.2s) takeoff window — the dip would eat the budget and trip the timeout. 80 steps
        # = 0.4s. (step = sim dt 0.005s, counted on physics substeps, so freq-independent.)
        takeoff_timeout_steps = 200         # 1.0s: room for dip + push
        grounded_grace_steps = 80           # MUST-LAUNCH grace (was 12 in base): the squat dip+hold (~50 steps)
                                            # must finish BEFORE _reward_grounded_jump penalizes "still grounded",
                                            # else it would punish the normal deep squat -> premature pop. 80 = a
                                            # full squat-hold-launch fits inside; only DITHERING (no launch) past it
                                            # is penalized. (timeout is 200, so penalty fires over steps 80-200.)
        # Squat-depth gate (countermovement) — REPLACES the failed time-window. successful_jump and
        # projected_peak are WITHHELD until this jump has dipped to <= squat_gate_height before
        # takeoff. So "don't dip" = no main rewards AT ALL (not just a 0.5s blackout) -> a real
        # countermovement is the only way to score. The time-window failed because gating rewards
        # for N steps didn't stop the policy from physically insta-popping; a depth gate ties the
        # reward to the dip itself. RSI air-drops exempt. (stance_window_steps removed.)
        # LAUNCH-ANGLE FIX (2026-08-12). forward_reach is paid every airborne step, so its integral is
        # (reach x hang time) and hang time is bought with vertical velocity -> the reward optimum sits
        # at ~51 deg instead of the ballistic 45 deg. Measured on model_4000: the launch SPEED is
        # identical to the old 4600 (2.68 m/s both) but the ANGLE is 62 deg vs 53 deg, and that alone
        # accounts for the whole 0.53 m vs 0.74 m clean-flight gap. Capping the paid window stops extra
        # hang time from paying. Swept: 0.45s->51 deg, 0.36->49, 0.34->47, 0.32->45 (ballistic optimum,
        # 88% of the term's magnitude retained), <0.30 only shrinks the driver. See _reward_forward_reach.
        # ⚠️ 2026-08-18 SET BACK TO 0 (= uncapped, the model_4600 behaviour). Same reason as the command
        # range above: the step-by-step plan starts from "4600 + joint limits only". The mechanism was
        # VERIFIED (peak 0.541 -> 0.500 = exactly the commanded cap, launch angle 62 -> 55 deg), but the
        # policy re-routed the saved hang time into a 77% LONGER ground creep (0.190 -> 0.336), so it is
        # not a clean win and does not belong in the baseline. Revisit after the observation work.
        forward_reach_window_s = 0.0

        soft_dof_pos_limit = 0.9            # was 1.0 (no margin = penalty only AT the hard limit = useless).
                                            # 0.9 -> dof_pos_limits starts penalizing in the last 10% before the
                                            # hard URDF limit, so the over-deep squat stops before jamming the "wall".
        squat_gate_height = 0.24            # must dip base to <=0.24m (idle ~0.31) to unlock jump rewards
        successful_jump_min_peak_height = 0.40  # was 0.30: a ~0.34 "low pop" no longer counts as success
                                                # (= command floor 0.40; kills the low-jump shortcut)
        # REVERTED to True (the decouple test did NOT fix the collapse -- root was default_pos taxing the jump,
        # now fixed by converting default_pos to a reward). Back to the baseline: successful_jump =
        # stay-upright(binary) x height_score x landing-accuracy (couples precision with stability).
        success_use_velocity_score = True
        success_landing_min_score = 0.2     # REVERTED to safe (was 0.0 for the failed always-on test): the 0.2
                                            # floor keeps a stable-but-short jump earning the bonus (no give-up
                                            # death spiral). Isolate the soft clean_takeoff_bonus as the only change.
        # DECOUPLE success from the squat_qualified HOLD gate (which flickers under pure-torque noise ->
        # made succ oscillate 0.01-0.89 while flight/peak were stable). peak>=0.40 already guarantees a
        # real countermovement, so the gate is redundant FOR SUCCESS. It still gates the jump-REWARD chain.
        success_requires_squat_qualified = False  # REVERTED to safe (was True for the failed always-on test):
                                                  # isolate the soft clean_takeoff_bonus as the single new variable.
        # PRE-JUMP STANCE ANCHOR: scale the default_pos penalty (toward q_squat) UP only during the
        # ~jumping_state STAND (pre-jump + post-finish), so the policy holds a COLLECTED ready stance (feet
        # under body) instead of sprawling pre-jump and then needing a recovery STEP (which trips jump_replant).
        # The jump itself keeps the base default_pos weight (no extra tax). 1.0 = off. Read in _reward_default_pos,
        # gated post-discovery on _takeoff_omega_on (succ_ema>=0.80). WATCH noise_std (default_pos -0.7 ran away
        # once); if the stand still sprawls, raise; if discovery/jumping suffers, lower toward 1.0.
        default_pos_prejump_scale = 3.0
        # stand_no_takeoff penalty: ignore the first N env-steps of each episode so the SPAWN drop (init base
        # 0.42 -> natural rest ~0.30 free-fall, not a hop) is not penalized; the time-based jump reflex (the
        # real target) fires later (~ep90+), so 50 cleanly separates them. Read in _reward_stand_no_takeoff.
        stand_no_takeoff_grace = 50
        # RSI static deep-squat air-drop (the EXPLORATION piece): half the RSI envs start AT REST in the
        # deep squat + jumping, so value learns "deep-squat-at-rest = high return" (they're gate-exempt
        # and earn jump rewards from the dip). This plants V(dip) that the squat-depth gate then makes the
        # standing policy chase. (fe15103 tried this WITHOUT the gate and failed; now the gate backs it.)
        # RSI DISABLED (2026-06-05): rsi_prob 0.2 -> 0. RSI teleported 20% of envs INTO q_squat
        # (dof=q_squat) AND exempted them from the gate, so they grabbed the biggest rewards in the
        # system (takeoff_vz 15 / projected_peak 20 / successful 400) for pushing UP from a squat --
        # i.e. it demonstrated ONLY the JUMP half (push), never the FOLD half (stand->squat, it
        # teleports past it), training the shared policy toward popping. With RSI off, NO env earns
        # the jump chain until it actually folds-then-jumps, so stance_squat becomes the dominant
        # early reward (the squat is the only thing that scores) -- the intended guidance.
        rsi_prob = 0.0                    # was 0.2 (inherited). See note above. static_frac etc. now inert.
        # --- Olsen-2025 style RSI (stage 1 turns these on; defaults keep the old behaviour) ---
        rsi_stand_sweep = False           # True = init AT REST at a randomized standing height, deep-squat
                                          # biased early and BROADENING to the nominal stance as succ rises
                                          # (_rsi_stand_sweep_pose). Fixes the 2026-06-05 failure: the old RSI
                                          # was a FIXED teleport into q_squat WITH launch velocity, so it only
                                          # ever demoed the push and never the fold -> popping.
        rsi_gate_exempt = True            # False = RSI envs must earn the squat gate like everyone else.
        rsi_sweep_foot_height_lo = 0.10   # deepest init (== squat_foot_height)
        rsi_sweep_foot_height_hi = 0.30   # tallest init (== nominal stance)
        rsi_sweep_levels = 5
        rsi_static_frac = 0.5              # of the rsi_prob envs, half = static deep-squat, half = launch
        rsi_static_vel_z_min = -0.1       # near-rest vz at the squat bottom (slight down/up)
        rsi_static_vel_z_max = 0.3

        # ---- landing-stability borrows from the papers (Olsen 2025 / Atanassov 2025) ----
        # (4) tighten the success gate so a touch-down-then-topple is NOT counted as success.
        success_fallover_tilt = 0.4       # was 0.7 inherited (~46 deg). projected_gravity_xy above this
                                          # during the landing buffer cancels the pending success. 0.4 ~ 24 deg:
                                          # the 400-wt successful_jump now only pays for a landing that STAYS upright.
        # (2) landing_impact regularization knobs (see _reward_landing_impact):
        landing_impact_force_floor = 150.0  # N total vertical foot force below which no impact penalty (~standing weight)
        landing_impact_force_norm = 1500.0  # N normalizer; impact penalty saturates at (floor + norm)
        # (clean single jump) TERMINATE the episode on a load-phase foot RE-PLANT (stutter-step / run-up):
        # the jump must be ONE clean push (all feet leave together; no shuffling/stepping before takeoff).
        # WHY: as the dx curriculum pushed to 1.5-1.6 the policy invented a run-up (squat_qualified eroded
        # 0.90->0.73, play showed it stutter-stepping to build momentum) which also rotted the near commands
        # (cmd 1.0 undershot to 0.72 at model_9700 vs 0.98 at the clean model_5000). Forbidding the re-plant
        # makes the only way to reach far a clean push -> the dx curriculum SELF-LIMITS at the clean-jump
        # range (no artificial dx_final cap). Pure termination: a stutter loses the whole jump reward, and a
        # clean-but-short jump still earns more, so terminating is never an escape hatch.
        clean_takeoff_terminate = False     # HARD gate OFF (user): zeroing the jump-reward chain on a re-plant
                                            # killed discovery (robot couldn't learn to jump). Replaced by the
                                            # SOFT clean_takeoff_bonus reward (clean pays more, messy allowed).
                                            # =False also makes the _squat_deep_enough ~jump_replant gate inert.
        clean_takeoff_min_step = 0          # EXPERIMENT (user): 60000 -> 0 = LITERAL always-on, the gate bites
                                            # from step 0. Every re-planting jump forfeits the WHOLE jump-reward
                                            # chain from the very first iter -> the policy must find a CLEAN
                                            # single-push jump from scratch, never building a replant habit.
                                            # ⚠️ HIGH discovery risk (code's note: early failed pushes also
                                            # re-plant): with no early grace window, messy from-scratch attempts
                                            # all re-plant -> jump rewards ~0 -> the robot may sit in the squat
                                            # and never jump. WATCH squatQ/jump_flight in the first ~iter150; if
                                            # flight stays ~0 (no jumping), discovery died -> raise min_step.
        # CLEAN-LANDING (user request): no small hop / shuffle-step after touchdown -> ONE clean settle. Once
        # all 4 feet HOLD contact for clean_landing_plant_hold steps (skips the impact chatter), any foot
        # lifting is penalized per-step by _reward_clean_landing (weight `clean_landing` in scales). PENALTY,
        # not termination (keeps the successful_jump bonus). Gated post-discovery (succ-latch _takeoff_omega_on).
        clean_landing_plant_hold = 15       # consecutive all-4-contact steps (~0.075s) to latch "settled" before watching for re-lift
        landing_pitch_extra = 5.0           # EXTRA pitch-leveling multiplier on prelanding+landing (see _reward_pitch_level):
                                            # the whole-cycle pitch term is diluted by the long level cruise + the fast post-tumble
                                            # termination, so it barely presses the touchdown. At 5.0 the descent/touchdown pitch is
                                            # penalized (1+5)x -> land PARALLEL to the ground, all four feet together.
        jump_pitch_extra = 12.0            # EXTRA pitch penalty across ALL JUMP PHASES (load->push->flight->landing; gated on
                                            # the succ-latch). ROOT of the front-first landing: the body is already nose-down IN
                                            # THE AIR (launches tilted; takeoff_omega froze the rotation so it stays tilted). A
                                            # small per-step penalty did NOT change it (the lean buys reach), so penalize the tilt
                                            # HARD ((1+12)x EVERY jump step, +landing_pitch_extra on top at landing) -> the body
                                            # must be LEVEL the whole jump -> forces a level push/launch. Tune: still tilted ->
                                            # raise (15/20); if level flight shortens the jump (dx drops) the lean bought reach -> ease.

        class scales(GO2OmniJumpCurriculumTorqueCfg.rewards.scales):
            # ---- proven jump-driving stack inherited UNCHANGED ----
            #   takeoff_vertical_velocity=10, projected_peak=15, successful_jump=300,
            #   orientation=-1.6, collision=-3.0, default_pos=-0.3, default_hip_pos=0.3, ...
            # ---- landing layer (new) ----
            tracking_linear_velocity = 0.0   # was 0.5: commands[0:2] is now meters, not m/s
            tracking_angular_velocity = 1.5  # 0.5 -> 1.5: reverse-engineer of earned 0.027 showed wz~0.28 rad/s
                                             # (~12 deg drift/jump) = only half-damped. 3x to tighten the hold.
                                             # Still << main jump rewards (peak25/vz15); earned ~0.08. ACTIVATE (was 0):
                                             # OmniNet-style yaw-rate damping to hold heading.
                                             # ang_vel_damp_zero_command=True (below) keeps it active at zero yaw cmd
                                             # so the error term = wz^2 = damp spin during flight -> fixes the heading
                                             # drift. Kept WELL below the main jump rewards (peak25/vz15/landing20); it's
                                             # a stabilizer. Stage2: open commands[2] -> same term becomes turn-tracking.
            # ⚠️ REVERTED 88 -> 60 (2026-08-12, run Aug12_17-36-58). 88 peaked HIGHER than anything before
            # it -- at iter 800: clean_reach 0.505, reliable_reach_dx 0.729, creep 0.171 (vs 0.336 on the
            # window run) -- and then COLLAPSED at iter ~1600 and never recovered: value_function loss
            # 0.130 -> 0.278 (4x the 0.068 of the joint-fix run), squat_qualified 0.94 -> 0.68,
            # successful_jump 0.93 -> 0.55, reliable_reach_dx 0.62 -> 0.24, then 3400 iterations of
            # thrashing. That is the value-variance failure the ">30 single weight" rule warns about,
            # and the caution comment below called it in advance. The other half of that change (command
            # range 0.5-1.5 -> 0.4-1.2) is KEPT: it is what brought creep down from 0.336 to 0.13-0.17
            # and noise_std from 0.159 to 0.10, and at iter 800 it was strictly better than the window
            # run on every axis. So keep the command fix, drop the weight bump.
            # The original 60 -> 88 rationale, for the record: capping the payment window cut this
            # term's earned value 31% (0.913 -> 0.633) and its share 35.8% -> 30.1%, weakening the one
            # distance term a ground shuffle cannot satisfy. That side effect is real but is NOT worth
            # paying for with a collapse -- revisit via the term's SHAPE, not its scale.
            forward_reach = 60.0             # 保留强的(user, 2026-07-04): option-1 eval欠程的真凶是 PROBE(喂不可能命令)不是
                                             # forward_reach. probe关了+漏算失败修了(dx_env诚实=命令都够得到)后, forward_reach强是好事:
                                             # 推策略把够得到的命令跳准跳足, 精度奖励(projected_landing/landing_position)防过冲, 平衡在正好落点.
                                             # DOMINANT driver so the policy pushes HARD to reach far. WATCH: if it trades
                                             # away HEIGHT (peak drops) or PRECISION, ease back to 25; if unstable, 20.
                                             # distance-progressive EFFORT reward (see _reward_forward_reach). Rewards
                                             # the projected FORWARD reach (absolute m, capped at the command) -> jumping
                                             # FARTHER pays MORE + far command > near command, DECOUPLED from precise landing.
                                             # Fixes the chronic give-up: once a far command is hard to HIT precisely, all the
                                             # precision rewards go hit-or-miss and the policy abandons the big jump; this keeps a
                                             # STABLE positive signal for trying-its-best/reaching-far so it never gives up. Strong
                                             # (>= projected_peak 20) so DISTANCE outranks HEIGHT. TUNE vs precision rewards if it
                                             # overshoots near commands or starves precision.
            four_leg_push = 0.0             # DISABLED (user): it blew the critic (value_loss 0.18->4.0 @iter1031) and
                                             # never worked (rew flat ~0.017). _reward_four_leg_push still exists (push-up-to-
                                             # target per ON-GROUND leg, rear may dominate, front-first allowed) -> re-enable by
                                             # setting weight only AFTER re-verifying the "rear idle" premise with the FIXED
                                             # torque_diag (the old "rear thigh ~0.3" finding used the broken pd0/general_scale=1 eval).
            projected_landing = 15.0         # 10 -> 15: BOOST. USER PRINCIPLE: jump-DISTANCE/accuracy rewards must OUTRANK jump-HEIGHT
                                             # rewards. After projected_peak 25->20, height earned ~0.60 (pp 0.38 + takeoff_vz 0.22) still
                                             # exceeded distance ~0.43 (landing_position + this), so raise the distance side above height.
                                             # The earlier 20->10 HALVING was to curb a "precise-but-topple" farm -- but stability is now SOLVED
                                             # (succ 0.90 via landing-focused pitch), so the topple risk is gone and boosting AIM is safe; it now
                                             # drives the forward launch toward the target. Discovery-safe (gated on a real jump peak>=0.40).
            projected_peak = 20.0            # 25 -> 20: modest OVERALL height trim. Run trend showed the policy FARMS height (projected_peak
                                             # rose 0.30->0.50 while landing accuracy collapsed to ~0 once far accuracy got hard) -- height is
                                             # the biggest reward AND paid regardless of landing, so the policy dumps accuracy for it. A mild 20%
                                             # cut eases that without gating. NOTE: an earlier 25->15 broke discovery, but that was BUNDLED with
                                             # base_ang_vel_xy -0.4 (the real flight-penalty killer), now reverted to -0.15 -> a mild 20 should keep
                                             # discovery. If 20 is too mild (still farms height / 1.1 plateau holds) go lower or add a non-vanishing
                                             # far-accuracy pull. WATCH iter~500 flight recovers; if flight 0, 20 is still too low -> revert to 25.
            successful_jump = 1000.0          # Jun23_01-23-30 baseline (reverted). Sparse so weight is big but earned
                                             # modest (~0.25; also graded by height_score). Coupled to landing accuracy
                                             # via _get_successful_jump_velocity_score (success_landing_min_score floor).
            landing_position = 8.0           # 5 -> 8: BOOST (with projected_landing 15) so jump-DISTANCE/accuracy OUTRANKS height
                                             # (user principle). DENSE over the landing buffer (~150 steps, fixed touchdown xy). Target:
                                             # distance earned (landing_position + projected_landing ~0.67) > height (~0.60). Discovery-safe
                                             # (gated on a real jump peak>=0.40 -> can't be farmed by standing). VERIFY the earned balance in
                                             # the retrain (height < distance); if height still wins, boost these more / trim height further.
            # ---- Stage2-ready: DISABLE takeoff_direction (was inherited 3.0) ----
            # takeoff_direction = vz/‖v‖ rewards a PURELY VERTICAL takeoff — the only Stage1-specific
            # reward. It is redundant at command 0 (takeoff_vz + projected_landing already give the
            # ballistic vertical+horizontal target) and FIGHTS any directed jump once landing_stage=2
            # opens dx/dy (it would penalize the horizontal velocity you NEED to reach the target).
            # Removing it now makes the whole stack direction-general: switching to Stage2 = just open
            # the command ranges, zero reward surgery. Behaviour stays in-place while commands[0:2]=0.
            takeoff_direction = 0.0
            # ---- MUST-LAUNCH: penalize "commanded to jump but still grounded after the squat window" ----
            grounded_jump = 0.0              # DEACTIVATED: made it WORSE. The policy isn't "refusing" to jump --
                                             # it physically can't push as hard once PD fades (a CAPABILITY problem,
                                             # not motivation), so forcing a launch only produced worse weak jumps +
                                             # extra penalty. The real lever is the PD-fade slowdown (growth.x0), below.
            # ---- MERGED takeoff launch: velocity-VECTOR match (height + distance in one), replaces vertical-only ----
            takeoff_vertical_velocity = 0.0  # OFF: superseded by takeoff_velocity_match (which == it at dx=0)
            takeoff_velocity_match = 15.0    # reward takeoff velocity matching the ballistic launch to (landing
                                             # point + apex height). CAUSE-side "jump FAR and HIGH" driver — the
                                             # closeness rewards can't push reach (diminishing returns at undershoot).
                                             # = old takeoff_vz weight (15). At dx=0 it reduces to takeoff_vz (safe).
            launch_pitch_toward_vel = 0.0    # RE-DISABLED (2026-07-11, FALSIFIED = 6th dead lever). At weight 10 it DID
                                             # pitch the body nose-up (0%->100% nose-up, align 0.40->0.70) but the take-off
                                             # speed COLLAPSED 2.25->1.08 m/s and reach fell to ~0.35 m: a nose-up attitude
                                             # is mechanically incompatible with a strong downward push. The body stays level
                                             # because that IS the strong-launch posture. Posture proxy, like leg_extension/util.
            # ---- structurally-inert rewards removed ----
            joint_angle_loaded = 0.0         # was 0.4: phase_loaded (jumping & ~taken_off & vz<=0) almost never
                                             # fires — the policy pre-squats during idle and pops straight up on
                                             # command, so there is no in-jump squat-down window. Contributed 0.0000.
                                             # SUPERSEDED by stance_squat below (same goal, NO vz<=0 dead-loop gate).
            # ---- countermovement: stance-squat shaping (the piece we were missing) ----
            stance_squat = 3.0               # was 1.5: now the PRIMARY early driver (RSI off -> jump chain locked
                                             # until folded, so this is the main thing firing early). pose-based
                                             # exp(-|dof-q_squat|/sigma). SHAPES the dip (how to get down); paid while loading and not yet at
                                             # squat_gate_height, then stops. The squat-depth gate on successful_jump/
                                             # projected_peak is the real forcing function (no dip -> no main rewards).
                                             # Farm-safe: stops at the gate, and squatting-without-jumping earns no
                                             # successful_jump anyway. (history: 0.5/2.0 weight-only + time-window all failed.)
            # ---- air-time + stay-planted boosts (user) ----
            clean_takeoff_bonus = 0.0        # DELETED (2026-07-11): reward-distribution audit showed it was INERT
                                             # (contributed 0.0001 = 0.0% -> never actually firing). Removed.
            stand_no_takeoff = 0.0           # DELETED (2026-07-11): jump-only config (jump_command_range=[1,1]) never
                                             # commands stand, so this never fires (contributed -0.002). Removed.
                                             # (was -5.0; only fired at cmd4<=0.5 which the jump-only config never samples.)
            all_feet_airborne = 3.0          # 2.0 -> 3.0: more air-time pressure. Gated (squat_deep + height_progress)
                                             # so it can't be farmed by a tucked sprawl. The policy currently UNDERSHOOTS
                                             # the commanded apex (peak ~0.50 vs cmd ~0.55) -> this pushes it to the FULL
                                             # commanded height -> longer flight -> more reach when vx is calf-capped
                                             # (range = vx*T). MODERATE on purpose: over-boosting pushes jumping ABOVE
                                             # the command and fights projected_peak. Watch mean_peak_height + hit_rate.
            maintain_contact = 0.3           # 0.10 -> 0.3: POSITIVE "four feet planted when not airborne" — the incentive
                                             # complement to clean_landing for the post-touchdown shuffle (lifting a foot
                                             # in the settle now costs this). MODERATE: a big value also pays the pre-jump
                                             # STAND -> could make "don't jump" comfy (discovery risk). Watch jump_flight_rate.
            landing_stability = 1.0          # RE-ENABLED to BRAKE the landing momentum: per-step trace showed the
                                             # robot lands at vx~1.3 m/s, bounces (all feet off, +0.06m) and coasts
                                             # ~0.30m forward to a stop (the "post-landing slide"). This rewards LOW
                                             # base velocity during the landing buffer -> absorb/stop on the spot.
            # NOTE: landing_stability_lin_sigma / _ang_sigma live in `class rewards` (NOT here in scales) — the
            # reward fn reads cfg.rewards.<name>. They were MISPLACED here, so cfg.rewards.<name> fell back to the
            # default 0.25 -> exp(-2.5^2/0.25)~0 -> the brake had ZERO gradient (verified: stab~0.000, slide 0.7m).
            # clean_landing REMOVED (ineffective: detector never armed -> reward ~0). Post-landing slide handled by
            # landing_stability (brake momentum) + disable_jump_on_landing (no commanded re-jump); error obs real-time.
            # ---- four-foot contact-timing sync (penalty on staggered takeoff/landing) ----
            foot_contact_sync = -4.0         # was -3.0: STRENGTHEN four-foot takeoff/landing sync (less body tilt
                                             # at touchdown). config note: if peak drops too much, back off.
                                             # penalize 1-3 feet on the ground during the takeoff push / landing
                                             # window -> all four feet leave & touch down TOGETHER (less body tilt).
                                             # active = (jumping & ~taken_off) | landing, so this tightens BOTH
                                             # liftoff and touchdown timing. COST: stronger sync caps peak a bit
                                             # (-2.0 already 0.576->0.539); if still uneven go -4.0, if peak drops
                                             # too much back off. (replaced the old joint-based pushoff_leg_sync.)
            # ---- kill in-air flailing (PD faded -> RL flails legs in the ~unconstrained air phase) ----
            aerial_dof_acc = -3e-6           # was -1e-6: too weak. Air joint-accel actually ~180 rad/s^2 (flailing);
                                             # air pose is a near-zero-gradient dim (peak/airborne rewards ignore leg
                                             # pose) so RL leaves it noisy. x3 so "flail vs tuck" actually moves the
                                             # return. Watch it doesn't over-damp the necessary tuck/extend.
            # ---- general joint-accel smoothness: REDUCED (overrides inherited -2.5e-7). It competes with the
            #      explosive pure-torque pushoff; halving frees the jump. aerial_dof_acc kept at -3e-6 (unchanged, per call).
            dof_acc = -1.25e-7
            # ---- action_rate: RESTORED to -0.03 (the off=0.0 test confirmed action_rate was NOT the collapse cause
            #      -- it still collapsed at 0). Back to -0.03 for anti-fidget. (Root cause was default_pos taxing the
            #      jump -> now fixed by converting default_pos to a reward.) Phase-gate later if it damps the burst.
            action_rate = -0.03
            # ---- pose-shaping joint_angle_* REMOVED (cleanup, audit): each earned ~0 (robot never reached
            #      q_air/q_pre/q_ground) = dead weight. Landing ATTITUDE now held by orientation + foot_contact_sync
            #      (strengthened below); landing-POINT by projected_landing + landing_position (kept / revived).
            joint_angle_aerial = 0.0
            joint_angle_prelanding = 0.0
            joint_angle_landing = 0.0
            # ---- post-PD pose-holding (rear legs drifted once PD faded to 0) ----
            default_pos = -0.5              # HALVED penalty (was -1.0). At -1.0 it was the DOMINANT term (-0.81/s =
                                             # 57% of ALL penalties) and TAXED the jump (a jump deviates from the pose
                                             # target) -> jumping net-negative -> policy collapsed to NOT jumping (Jun21
                                             # runs: best ~iter500, collapse iter700-1100). Halving cuts the jump tax
                                             # ~in half (-0.81 -> ~-0.4) so jumping stays net-positive, kept as a PENALTY
                                             # (simpler than a reward; no standing-pose reward to make not-jumping comfy).
                                             # Zeroed during push-off / squat-down. NOTE: memory says -0.7 once caused
                                             # noise runaway -> watch noise_std; raise back if the anchor gets too loose.
            # 2026-07-11 LATERAL: default_hip_pos is now COMMAND-CONDITIONAL (see _reward_default_hip_pos override in the
            # landing env): full anti-slide hip lock for forward commands, relaxed toward 0 as |d_y| grows so a side jump can
            # abduct the hips. Its knob default_hip_pos_lat_ref lives in `class rewards` (NOT here in scales).
            default_hip_pos = 2.0            # 1.0 -> 2.0 (user 2026-07-05): 髋外展/内收 splay 差, 强锁髋(4个hip-abduction关节)到 default.
                                             # [0.3 -> 1.0 史]: the policy slid the front feet INWARD (hip adduction) to shuffle
                                             # forward momentum (the stutter/run-up morphed into a SLIDE once the re-plant
                                             # termination forbade stepping). default_hip_pos keeps the 4 hip-abduction joints
                                             # near default; at 0.3 it earned only ~0.05 (hips drifting ~0.46 rad) -- too weak to
                                             # hold them. Raised so deviating forfeits a meaningful reward -> legs stay vertical
                                             # in the frontal plane (no inward collapse). Safe: a clean forward jump is sagittal
                                             # (thigh/calf) and never needs hip abduction. Tune up (1.5-2.0) if the slide persists;
                                             # if it persists even then it's pure ground-slip (not hip) -> add a foot_slip penalty.
            orientation = -3.5               # -3.0 -> -3.5 (DISCOVERY-SAFE: -4.5 + the strong default_pos made not-jumping
                                             # too comfortable from scratch, Jun09_11-29-05). Mild strengthen of the level-body
                                             # hold (late training showed g_xy^2 creeping 0.017->0.038 as the policy traded
                                             # attitude for jump magnitude). Vertical (Stage1) jump wants body level throughout.
                                             # the main landing-stability lever after joint_angle_landing removed. (pitch also via pitch_level.)
            # ---- (1)+(2) landing stability from the papers: stop "lands then flips" ----
            base_ang_vel_xy = -0.15          # REVERTED -0.4 -> -0.15: -0.4 penalized the MESSY exploratory flight so hard that "don't jump"
                                             #     won -> discovery died (flight 0 at iter526). -0.15 is the discovery-proven value. The nose-down
                                             #     ROTATION is a LATE concern -> if needed, re-add stronger damping GATED to post-discovery, not step 1.
                                             # (1) PENALTY on base roll/pitch angular velocity in flight+landing
                                             # (Olsen ϕσ(‖ω‖) / Atanassov "track zero ω after landing"). We had NO
                                             # ω damping -> body tumbled into touchdown. Penalty (not bell kernel) so
                                             # it bites; penalizes spin RATE not airborne time -> clean high jump unhurt.
                                             # THE knob: still flipping -> more negative; jumps get stiff/low -> back off.
            dof_pos_limits = -5.0            # ENABLE (was 0/off): penalize joints folding past the soft limit
                                             # (soft_dof_pos_limit=0.9 below = last 10% before the hard URDF limit).
                                             # Fix for the over-deep squat (base ~0.13) jamming the knees to the
                                             # "wall" + stalling, which also dropped peak. Stops the dip ~10% short
                                             # of the hard limit -> smoother push, should recover height. Tunable.
            landing_impact = 0.0             # DISABLED (user: "没什么用"). Was -2.0 (Olsen soft-impact). Landing
                                             # cushion now handled by landing_stability (brake, sigma fix) + the
                                             # touchdown dynamics; this term wasn't earning its keep. (Aside: it also
                                             # risks suppressing height -- softer impact = jump lower -- which fights
                                             # our distance goal.) Re-enable only if touchdowns start slamming.
            pitch_level = -6.0               # -4.5 -> -6.0: further STRENGTHEN (preemptive vs nose-dive when we push
                                             # height higher; back off if the jump gets stiff/peak drops). Nose-dive is the high-jump
                                             # failure mode -- Jun05_23-55-11 crashed at pitch 0.68). PITCH-specific
                                             # attitude penalty (projected_gravity_x^2) over the whole
                                             # jump. Fixes the persistent nose-down ("head-heavy") tilt that the
                                             # symmetric orientation (-2.0) is too weak on. Stacks on orientation ->
                                             # pitch weighted ~2.5x roll during the jump. THE knob: still nose-down ->
                                             # more negative; jump gets stiff/weak or peak drops -> back off.

    class logging(GO2OmniJumpCurriculumTorqueCfg.logging):
        # Decluttered TERMINAL print: drop the redundant metrics (jump_landing_rate / jump_completed_cycles
        # ≈ jump_flight_rate). EVERYTHING still goes to tensorboard/wandb -- this only filters the terminal.
        print_episode_keys = [
            k for k in GO2OmniJumpCurriculumTorqueCfg.logging.print_episode_keys
            if k not in ("jump_landing_rate", "jump_completed_cycles")
        ] + [
            "rew_takeoff_velocity_match",  # merged launch driver (jump far+high) — watch vs dx_max advancing
            "rew_launch_pitch_toward_vel", # nose-velocity alignment during ascending -> rear-leg push
            "rew_projected_landing",
            "rew_landing_position",
            "rew_foot_contact_sync",
            "rew_stance_squat",     # countermovement shaping — watch vs successful_jump_rate
            # inherited-active but missing from the parent's print list — surface them
            "rew_base_ang_vel_xy",   # (1) flight+landing roll/pitch ω damping — the anti-tumble lever
            "rew_landing_impact",    # (2) touchdown force-spike penalty — cushion vs slam
            "rew_pitch_level",       # pitch-specific tilt penalty — fix persistent nose-down
            "rew_tracking_angular_velocity",  # OmniNet yaw-rate damping (hold heading) — watch it stays < jump rewards
            "rew_dof_pos_limits",    # joint-limit penalty — watch it shrinks as the over-deep squat stops jamming
            "squat_qualified_rate",  # frac of takeoffs preceded by a HELD squat; compare to jump_flight_rate
            # ---- distance curriculum (watch these to see the dx ramp progress) ----
            "landing_dx_max",            # per-env 双向课程: 全局最大上界 (最强 env)
            "landing_dx_mean",           # per-env 双向课程: 群体平均上界 = 真实纯力矩能力 (★盯这个诚实收敛~0.6-0.7)
            "landing_dx_min",            # per-env 双向课程: 最弱 env 上界
            "landing_dx_stable_cum",     # CUMULATIVE far-band stable-hit the gate reads (>= thr AND enough samples -> advance)
            "landing_stable_hit_uniform",# 又准又稳 over all dx (uniform)
            "landing_hit_rate",          # accuracy (ignores stability) — ⚠️ uniform/near-inflated, NOT capability
            "landing_farband_hit_smooth",# ★ HONEST 远端掌握度 (平滑, 抗 far_n~0.1 噪声) — 盯这个, 别信 hit_rate/dx_max
        ]

    class test(GO2OmniJumpCurriculumTorqueCfg.test):
        vel = GO2OmniJumpCurriculumTorqueCfg.test.vel.clone()
        vel[0] = 0.0   # Stage 1: land in place (dx=0). Set vel[0]>0 to play directed jumps.
        vel[1] = 0.0
        single_jump_play = True    # 单跳: play 跳一次就停站立 (撤回连续跳的 False)


def _mirror_obs_permutation(history_length, frame_dim, extras_dim):
    """LEFT-RIGHT mirror map for the STACKED layout [ frame(49) x history_length | extras(32) ].

    ppo.py tiles a single frame permutation `frame_stack` times, which only works when the whole obs
    is a uniform stack. Ours has trailing single-frame extras, so we hand it the COMPLETE 1012-dim map
    and leave frame_stack=1 -- the tiling below is done here, where the layout is known.

    Mirroring is about the sagittal plane: v_y / w_x / w_z / g_y / lateral target error flip sign, and
    the legs swap FL<->FR, RL<->RR with the hip (abduction) joint flipping sign. WARNING: `torques`
    flips the hip sign (it is a signed torque) but `motor_fatigue` does NOT -- it integrates |tau|, a
    magnitude. Blocks/signs were cross-checked term by term against the flat 69-dim map in
    go2_omnijump_torque_config.py and agree everywhere EXCEPT:

      * projected_gravity. The parent map says `-6, 7, 8` = flip g_x, keep g_y. That is backwards: a
        sagittal reflection preserves the x and z components and flips y (the mirror image of a
        nose-down robot is still nose-down -> g_x must NOT flip; the mirror image of a robot leaning
        left leans right -> g_y MUST flip). We use the correct `+g_x, -g_y, +g_z` here. The parent's
        version is a real bug that every flat run inherited -- left alone on purpose, since changing it
        silently retrains every other task; fix it as its own step if you want it.
      * the third command slot. In the parent it is commands[2] (a yaw-rate command, sign-flipped);
        here that slot is the landing error's hard-coded ZERO third channel, so its sign is moot.
    """
    def enc(idx, sign):
        # ppo.py reads the index as int(abs(v)) and the sign as np.sign(v), so index 0 needs a stand-in.
        return sign * (idx if idx != 0 else 0.0001)

    def legs(base, flip_hip=True):
        """12-vector of (hip, thigh, calf) x (FL, FR, RL, RR) at `base` -> mirrored source/sign."""
        src = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        sgn = [-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1] if flip_hip else [1] * 12
        return [(base + i, g) for i, g in zip(src, sgn)]

    frame = (
        [(0, 1), (1, -1), (2, 1)]                  # base_lin_vel      (v_y)
        + [(3, -1), (4, 1), (5, -1)]               # base_ang_vel      (w_x, w_z)
        + [(6, 1), (7, -1), (8, 1)]                # projected_gravity (g_y)
        + legs(9)                                  # dof_pos
        + legs(21)                                 # dof_vel
        + [(34, 1), (33, 1), (36, 1), (35, 1)]     # foot contact       FL<->FR, RL<->RR
        + legs(37)                                 # previous action    (= act_permutation)
    )
    extras = (
        [(0, 1), (1, -1), (2, 1)]                  # yaw-frame landing error (lateral flips)
        + [(3, 1), (4, 1), (5, 1), (6, 1)]         # cmd height, cmd4 (jump toggle), height obs x2
        + legs(7)                                  # torques        (signed -> hip flips)
        + legs(19, flip_hip=False)                 # motor_fatigue  (magnitude -> NO sign flip)
        + [(31, 1)]                                # pd_prior_alpha
    )
    assert len(frame) == frame_dim, f"frame perm {len(frame)} != {frame_dim}"
    assert len(extras) == extras_dim, f"extras perm {len(extras)} != {extras_dim}"

    perm = []
    for k in range(history_length):
        perm += [enc(i + k * frame_dim, g) for i, g in frame]
    off = history_length * frame_dim
    perm += [enc(i + off, g) for i, g in extras]
    return perm


class GO2OmniJumpLandingTorqueCfgPPO(GO2OmniJumpCurriculumTorqueCfgPPO):
    class policy(GO2OmniJumpCurriculumTorqueCfgPPO.policy):
        # Step H (final): τ_comp is a DETERMINISTIC independent head (12 outputs), NOT part of the PPO
        # action (num_actions stays 12 = τ_jump). BC hits only comp_head; PPO over τ_jump = single-head.
        aux_head_dim = 12

    class algorithm(GO2OmniJumpCurriculumTorqueCfgPPO.algorithm):
        # sym_loss BACK ON (2026-08-21) -- Step 1b, and it is a BUG FIX, not a tuning knob.
        # Step 1 turned it off because ppo.py can only tile a frame permutation across a uniform stack and
        # ours is [49 x 20 | 32 extras]. That made Step 1 secretly TWO changes, and the second one cost us
        # heading discipline: with the mirror constraint gone the policy learned a one-sided SPIN --
        # measured on Aug18_23-09-30/model_4700, every jump turns the SAME way, 116 deg (dx0.5) / 126 (dx0.9)
        # / 150 (dx1.2), |dyaw|>20deg on 100% of landings, peak |wz| 7.8 rad/s. The flat baseline
        # (Aug08_16-35-49/model_4700, sym_coef=1.0) turns 1.1 deg with peak |wz| 0.45. Frame by frame the
        # yaw rate is built up ON THE GROUND during the push (yaw reaches -91 deg BEFORE takeoff) and the
        # flight merely integrates the conserved angular momentum -- which is why the airborne-only yaw damp
        # (tracking_angular_velocity + ang_vel_damp_airborne_only) cannot fix it: it acts where there is no
        # contact force to act with. The baseline has that same hole and stays straight, so the missing
        # ground-phase yaw penalty is what lets the spin GROW, not what causes it. The cause is the mirror.
        # No rsl_rl change needed: hand PPO the COMPLETE 1012-dim map (built above) and keep frame_stack=1
        # so it is not tiled again. This also settles "is sym_loss load-bearing for discovery?" in the other
        # direction: Aug18 discovered at iter650 with sym fully OFF.
        # ⚠️ OFF AGAIN (2026-08-25), and this time it is a CHOICE, not a limitation. The ablation
        # (Aug25_19-53-29, flat with sym off) showed the mirror loss is the wrong tool here: it is what was
        # holding the spin down, but it does it by banning ALL left-right asymmetry, and the same run says
        # that asymmetry is partly what buys reach -- flat sym-off tracked the distance command at
        # dAir +0.21..0.31 against +0.03 for flat sym-on, which does not track at all. `base_ang_vel_yaw_weight`
        # now forbids the SPIN specifically, with a gradient that grows instead of vanishing. So this run is
        # Aug18 + the yaw penalty = one variable against a known-good reference.
        # Everything needed to switch it back on is still here and tested: the 1012-dim mirror map above and
        # the runner's discovery gate. Flip sym_loss to True and restore sym_gate_metric to use it.
        sym_loss = False
        sym_coef = 1.0   # the TARGET the gate would ramp to, if sym_loss is switched back on
                         # ⚠️ this is the TARGET value: the runner holds sym_coef at 0 until the discovery
                         # gate opens, then ramps to this over sym_ramp_iters (see runner cfg, sym_gate_*).
        frame_stack = 1  # the map below is ALREADY full-length -- do NOT let ppo.py tile it
        obs_permutation = _mirror_obs_permutation(
            GO2OmniJumpLandingTorqueCfg.env.history_length,
            GO2OmniJumpLandingTorqueCfg.env.num_stacked_frame,
            GO2OmniJumpLandingTorqueCfg.env.num_single_extras,
        )
        # Step H (final): τ_comp is OUT of the PPO action, so act_permutation stays 12-dim (inherited
        # = single-head). BC loss weight for the deterministic comp_head:
        bc_coef = 1.0
        entropy_coef = 0.003   # 0.001 -> 0.003: MORE exploration. At 0.001 noise_std collapsed to ~0.04 -> the
                               # policy got too CONSERVATIVE (peak ~0.50, undershoots far) and plateaued; the old
                               # high+far run had noise ~0.39. 0.003 settles noise ~0.32 (memory) = that exploration
                               # level but STABLE (the runaway was 0.005 -> 0.86), and forward_reach now holds the
                               # floor so the bigger jumps don't degrade. WATCH noise_std: ~0.2-0.35 good; toward 0.5+
                               # = runaway -> drop to 0.002.
                               # Constant 0.005 (run Jun06_07-41-08) cracked the squat gate (squatQ->0.95) but then
                               # noise RAN AWAY to 1.33 -> degraded from iter3000, collapsed at 5400. Constant 0.003 was
                               # the opposite (noise 0.32 -> stuck). So: high early (discover/crack gate), low late
                               # (anneal = consolidate + kill the noise runaway). See entropy_coef_final.

    class runner(GO2OmniJumpCurriculumTorqueCfgPPO.runner):
        experiment_name = "go2_omnijump_landing_torque"
        run_name = "stage1_landing"
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        max_iterations = 5000    # 3000 -> 5000 (user 2026-07-16). PD fades early (~iter800), then plenty of room for the
                                 # pure-torque policy to consolidate + dx_max to evolve (with the safety-revert).
        # entropy_coef ANNEALS 0.005 -> 0.001 at entropy_anneal_iter (HARD STEP, on_policy_runner.py:129-133).
        # MOVED 2800 -> 500 for the real-Go2 actuator. The 0.005 START is the ONLY force pushing action_std UP;
        # with the weak real calf the precise squat-jump can't survive high noise, so noise_std running away
        # (0.36 -> 0.6-0.87) crashed every run at iter700-1100 (squatQ -> 0, value_loss -> 0). 2800 was tuned for
        # the OLD strong calf whose peak hit ~iter2500; the NEW calf cracks the squat gate by iter100 (squatQ 0.91)
        # and noise bottoms at ~iter300, so anneal at 500 LOCKS the early peak (~0.58) and kills the runaway BEFORE
        # it crosses ~0.55 and collapses the jump. (Raise to 600-800 if a fresh run discovers slower; data = iter100.)
        entropy_anneal_iter = 500
        entropy_coef_final = 0.001
        # sym_coef DISCOVERY GATE (2026-08-25). Step 1b's first run (Aug25_15-46-36, commit 4161ead,
        # verified clean) never discovered: squatQ 0 for 3400 iters, reward stuck ~1.2, value_loss 0.001,
        # peak 0.30 (below the 0.42 standing height = it only ever crouched). The mirror loss itself looks
        # innocent -- Loss/symmetry ran 0.0026-0.0036, the same size as in the flat baseline that discovered
        # fine (Aug08, sym_coef also 1.0) -- but the history obs moved DISCOVERY from iter 51 (flat) to
        # iter 614-711 (Jul31/Aug18), i.e. into the tail where noise_std has already decayed to ~0.09, and
        # 2 of 3 history runs got through on luck. So the mirror constraint is treated like every other
        # stability lever here: HELD OUT until the behaviour exists, then ramped in. Before the gate this
        # reproduces exactly the Aug18 condition (sym off), which discovered 2 times out of 2.
        # Loss/symmetry keeps being logged while the gate is shut (coef 0 = no gradient), so the spin can
        # still be watched as it develops. Tensorboard: Policy/sym_coef + Policy/sym_gate_ema.
        sym_gate_metric = None    # gate DORMANT while sym_loss=False (see algorithm). Set back to
                                  # "squat_qualified_rate" together with sym_loss=True.
        sym_gate_value = 0.5      # EMA of squat_qualified_rate that counts as "discovered"
        sym_gate_ema_alpha = 0.05  # ~14 iters to cross 0.5 once the raw metric pins at 1.0
        sym_ramp_iters = 300      # 0 -> sym_coef, gradual so the constraint never lands as a shock


# ============================================================================================
# STAGE 1 -- IN-PLACE JUMP (Atanassov 2025 pi_1)
# ============================================================================================
# WHY a separate task rather than "just set the command to 0" inside the stage-2 config:
#   The behaviour we want is ONE COHERENT PUSH. In an in-place jump that is the UNIQUE solution --
#   you cannot shuffle your way into the air -- so stage 1 needs no anti-stutter penalty, no gate
#   and no termination: the target behaviour is the only behaviour that scores. That is structurally
#   different from bolting a run-up penalty onto a task that also pays for creeping, which is what
#   every previous attempt did (clean_takeoff_terminate / run_up gate / run_up penalty -- all three
#   collapsed discovery, because they were applied while discovery was still fragile).
#
# Three changes, nothing else (the 25 reward weights are untouched -- they DEGENERATE correctly at
# dx=0: forward_reach is capped at cmd_dist -> ~0 and inert, projected_landing / landing_position /
# takeoff_velocity_match become "launch vertically and land where you left", the height terms are
# already what stage 1 wants):
#   1. dx = dy = 0
#   2. landing_anchor_jump_start -- without it the target is anchored at the SQUAT BOTTOM, which is
#      downstream of the creep, so the creep would be merely NEUTRAL instead of strictly negative.
#   3. RSI on, Olsen-style (at rest, height sweep, no gate exemption). Atanassov's ablation: "the RSI
#      is required for learning the jumping-in-place task. Without it, the agent converges to a local
#      optimum and fails to complete the task."
#
# Stage 2 = the normal go2_omnijump_landing_torque task, warm-started from this policy (same env,
# same obs/action dims). Atanassov's other ablation is the reason the warm start matters: "directly
# training the long-distance jump [even with RSI] also results in an early convergence to a standing
# behavior, which highlights the need for our curriculum strategy."
# TWO-RUN RECIPE (2026-08-12, replaces the in-run hand-off -- see stage2_auto for why):
#   1) stage 1, in place, to convergence:
#        python legged_gym/scripts/train.py --task=go2_landing_stage1 --headless
#      -> logs/go2_landing_stage1_inplace/<run>/model_5000.pt
#   2) stage 2, the normal forward task, warm-started from it with a FRESH PPO optimizer:
#        RESUME_FRESH_OPTIMIZER=1 python legged_gym/scripts/train.py \
#            --task=go2_omnijump_landing_torque --headless --resume \
#            --load_run=/ABSOLUTE/path/to/logs/go2_landing_stage1_inplace/<run> --checkpoint=5000
#      An ABSOLUTE --load_run works across experiment folders because get_load_path does
#      os.path.join(log_root, load_run), which discards log_root when load_run is absolute.
#   Known difference from the in-run hand-off: stage 2 starts a NEW env, so step_count restarts and
#   the PD scaffold fades in again from 0.5. That is left as-is -- it makes stage 2 comparable to the
#   from-scratch baseline, which also trains through the fade.
class GO2OmniJumpLandingTorqueStage1Cfg(GO2OmniJumpLandingTorqueCfg):
    class commands(GO2OmniJumpLandingTorqueCfg.commands):
        landing_disp_x_stage2 = [0.0, 0.0]     # in place: no forward component
        landing_disp_y_stage2 = [0.0, 0.0]     # in place: no lateral component
        landing_anchor_jump_start = True       # creep == landing error (only strictly negative at dx=0)
        landing_dx_curriculum = False          # no distance curriculum in stage 1
        # ---- AUTOMATIC STAGE-1 -> STAGE-2 HANDOVER (user 2026-08-07: one run, no checkpoint juggling) ----
        # Latched ONE-WAY on MEASURED stage-1 mastery: jumps reliably (succ EMA) AND takes off cleanly
        # (creep EMA). The clean takeoff is the whole point of stage 1, so it GATES the handover instead
        # of merely being logged. Both EMAs use the 0.99 smoothing already used for the omega latch.
        # ⚠️ OFF since 2026-08-12. The in-run hand-off was TRIED (run Aug12_04-16-44) and the result was
        # not "handed over too early", it was "the warm start is harmful here": value_loss spiked to
        # 0.558 at the switch but was back to 0.031 within 40 iters -- fully recovered -- and yet
        # clean_reach stayed at HALF the from-scratch run for the remaining 4800 iterations
        # (0.24/0.25/0.36 at iter 500/1000/4999 vs 0.50/0.57/0.52). The in-place policy is a strong
        # attractor: unlearning "push straight up" costs more than learning forward from scratch.
        # The one thing it did win: run_up_creep 0.085 vs 0.336, so the in-place task really does
        # suppress the shuffle -- at the cost of the forward skill.
        # Also note the gate itself was mis-designed: creep <= 0.05 is satisfied from the START of
        # stage 1 (the policy has not learned to shuffle yet, creep GROWS with training), so "low
        # creep" was never evidence of mastery, and succ >= 0.85 lands at ~iter 200 because jumping in
        # place is easy -- which is the whole point of stage 1. Only the step floor bound, and it
        # happened to also be ~iter 208. If this is revisited, gate on peak_height_error (0.19 -> 0.03
        # is a real skill signal) and hold it, and keep creep as a VETO, not a trigger.
        # Now the two stages are run SEPARATELY (Atanassov-faithful: own run, own reward set, fresh
        # PPO via RESUME_FRESH_OPTIMIZER=1); see the recipe in the class docstring below.
        stage2_auto = False
        stage2_succ_gate = 0.85          # successful_jump_rate EMA
        stage2_creep_max = 0.05          # run_up_creep EMA [m] -- "one coherent push", no shuffling
        stage2_min_steps = 20000         # floor on common_step_counter before any handover (a fluke early
                                         # EMA is not mastery). NOTE common_step_counter counts PHYSICS
                                         # SUBSTEPS (post_physics_step runs per substep in go2_torque), so
                                         # it is ~96/iter during warmup and ~48/iter at pure torque:
                                         # 20000 ~ iter 200-400. Deliberately a rough floor, not a schedule.
        # ranges the handover opens (== the stage-2 task's ranges)
        landing_disp_x_stage2_after = [0.4, 1.2]
        landing_disp_y_stage2_after = [-0.30, 0.30]

    class rewards(GO2OmniJumpLandingTorqueCfg.rewards):
        rsi_prob = 0.3
        rsi_stand_sweep = True
        rsi_gate_exempt = False

        # NOTE forward_reach keeps its weight 60 here on purpose. A zero scale is POPPED in
        # _prepare_reward_function, so the term would never be registered and could not be switched on
        # at the stage-2 handover. Instead `_reward_forward_reach` RETURNS ZERO while the stage-1 anchor
        # is active -- it has to, because it pays ABSOLUTE metres capped at cmd_dist =
        # |landing_target - takeoff_xy|, which under the jump-start anchor equals |creep|: "shuffle 0.5 m
        # forward, then fly 0.5 m BACK" would farm 0.5 x 60 while a clean in-place jump earns 0. Flipping
        # landing_anchor_jump_start at the handover therefore revives it automatically -- one flag.
        # (Every other distance term degenerates safely at dx=0: projected_landing / landing_position
        # score the ERROR from the anchor so creeping only hurts; takeoff_velocity_match is a bounded
        # [0,1] match, so a creep only changes which launch velocity is correct -- it cannot be farmed.)



class GO2OmniJumpLandingTorqueStage1CfgPPO(GO2OmniJumpLandingTorqueCfgPPO):
    class runner(GO2OmniJumpLandingTorqueCfgPPO.runner):
        experiment_name = "go2_landing_stage1_inplace"
        run_name = "stage1_inplace"
