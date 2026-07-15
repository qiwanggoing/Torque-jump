"""Landing-point tracking on top of the proven vertical-jump curriculum env.

See go2_omnijump_landing_torque_config.py for the full design rationale. In
short: inherit the validated GO2OmniJumpCurriculumTorque jump machinery
unchanged and add a thin landing layer (landing target, yaw-frame error
observation, dense projected-landing + sparse landing-position rewards).
"""

import math

import torch
from isaacgym.torch_utils import get_euler_xyz, quat_apply

from legged_gym.envs.go2.go2_omnijump_curriculum_torque.go2_omnijump_curriculum_torque import (
    GO2OmniJumpCurriculumTorque,
)
from legged_gym.envs.go2.go2_omnijump_landing_torque.go2_omnijump_landing_torque_config import (
    GO2OmniJumpLandingTorqueCfg,
)


class GO2OmniJumpLandingTorque(GO2OmniJumpCurriculumTorque):
    cfg: GO2OmniJumpLandingTorqueCfg

    # Inherit the proven jump-driving reward set and add the two landing terms.
    ACTIVE_REWARD_WHITELIST = GO2OmniJumpCurriculumTorque.ACTIVE_REWARD_WHITELIST | {
        "landing_position",
        "projected_landing",
        "forward_reach",     # distance-progressive EFFORT reward (farther = more), decoupled from precise landing
        "foot_contact_sync",
        "stance_squat",
        "base_ang_vel_xy",   # landing stability: flight+landing roll/pitch ω damping
        "landing_impact",    # landing stability: touchdown force-spike penalty
        "pitch_level",       # landing stability: pitch-specific tilt penalty
        "dof_pos_limits",    # penalize folding joints to the limit (over-deep squat hits the "wall")
        "takeoff_velocity_match",  # merged launch: takeoff velocity-vector match to (landing point + apex) — jump far+high
        "launch_pitch_toward_vel", # task-space launch alignment: reward body nose pointing along velocity vector during ascending
        "landing_stability", # ABSORB the landing: reward low base velocity during the landing buffer so the forward
                             # momentum (vx~1.3 at touchdown) is BRAKED, not bounced+coasted (the post-landing slide)
        "grounded_jump",     # MUST-LAUNCH penalty (negative weight): close the retreat-to-not-jumping escape ->
                             # force the policy to TRY ITS BEST to launch. Discovery-gated in _reward_grounded_jump.
        "four_leg_push",     # SYNC FOUR-LEG PUSH: reward EVEN vertical-GRF across the 4 feet during push so the
                             # idle rear legs (torque_diag: rear thigh ~0.24 throttle vs front ~0.93) pull their
                             # weight -> more total impulse -> farther. Discovery-gated in _reward_four_leg_push.
        "clean_takeoff_bonus",  # SOFT clean-takeoff: extra reward for a no-re-plant takeoff (clean pays MORE,
                                # messy still allowed) -- replaces the hard clean_takeoff_terminate that killed discovery.
        "stand_no_takeoff",     # HARD penalty: cmd4=0 (STAND) but all feet leave the ground (a hop) -> punish ->
                                # cmd4 becomes the real stand/jump switch. Gated post-discovery + grace (skips spawn drop).
    }   # clean_landing REMOVED (detector never armed -> ~0). Post-landing slide handled by landing_stability
        # (brake momentum) + disable_jump_on_landing (no commanded re-jump); error obs is real-time (no obs-hold).

    # Curriculum gate table requires an entry for every active reward. Curriculum
    # is disabled (one-stage), so the stage value only needs to exist; 0 = active
    # from step 1 alongside the rest of the regularisation/task stack.
    REWARD_START_STAGES = {
        **GO2OmniJumpCurriculumTorque.REWARD_START_STAGES,
        "landing_position": 1,
        "projected_landing": 1,
        "forward_reach": 1,
        "four_leg_push": 1,
        "clean_takeoff_bonus": 0,   # active from step 1 (soft positive bonus = discovery-safe)
        "stand_no_takeoff": 0,      # stage 0; real gate is _takeoff_omega_on inside the reward (post-discovery)
        "foot_contact_sync": 0,
        "stance_squat": 0,
        "base_ang_vel_xy": 0,   # active from step 1 (curriculum disabled = one-stage)
        "landing_impact": 0,
        "pitch_level": 0,
        "dof_pos_limits": 0,
        "takeoff_velocity_match": 0,   # active from step 1 (replaces takeoff_vertical_velocity)
        "launch_pitch_toward_vel": 0,  # active from step 1 (ascending gate inside reward)
        "landing_stability": 0,   # active from step 1 (override parent's stage 3, which never fires one-stage)
        "grounded_jump": 0,   # stage 0; the real gate is the succ-EMA latch inside _reward_grounded_jump
    }   # clean_landing REMOVED (see whitelist note above)

    # ------------------------------------------------------------------ #
    # Buffers
    # ------------------------------------------------------------------ #
    def _init_buffers(self):
        super()._init_buffers()
        # World-frame desired landing xy, set each reset from spawn + commanded
        # displacement. Initialised to spawn so step-0 obs is well-defined.
        self.landing_target = self.root_states[:, :2].clone()

        # ANTI-CHEAT distance anchor: base xy at the SQUAT BOTTOM (deepest load, base_z lowest) of the
        # current jump. Landing distance is measured squat-bottom -> land (NOT spawn -> land), so the
        # ground creep during "stand -> squat" cannot farm distance, while the normal push-off travel
        # (squat -> liftoff) still counts. Updated during loading in _update_jump_state; landing_target
        # is locked to squat_root_xy + dx at takeoff. Reset to current xy at each _start_jump.
        self.squat_root_xy = self.root_states[:, :2].clone()

        # Stage 2: widen the displacement ranges that the parent's
        # _resample_commands draws commands[0:2] from. Stage 1 leaves them [0,0]
        # (land in place -> behaviour identical to the proven vertical jumper).
        stage2 = int(getattr(self.cfg.commands, "landing_stage", 1)) >= 2
        # Curriculum only ever runs in Stage 2 (so the EMA buffers below are always initialised
        # before _update_dx_curriculum reads them).
        self.landing_dx_curriculum = stage2 and bool(getattr(self.cfg.commands, "landing_dx_curriculum", False))
        self.landing_dx_max = float(getattr(self.cfg.commands, "landing_dx_final", 0.40))
        if stage2:
            self.command_ranges["lin_vel_y"] = list(self.cfg.commands.landing_disp_y_stage2)
            # Reduce time spent IDLING. Each resample (every resampling_time s) draws commands[4] and
            # STANDS if it is <= jump_command_threshold (0.5). The default range [0,1] -> 50% stand
            # per resample. Narrow it so the per-resample stand prob ~= 0.1, so the robot jumps almost
            # every resample (the IMPORTANT standing — returning to a stable stand AFTER landing — is
            # still trained in every jump episode's post-landing buffer).
            jc = getattr(self.cfg.commands, "jump_command_range", None)
            if jc is not None:
                self.command_ranges["jump_command"] = list(jc)
            if self.landing_dx_curriculum:
                # DISTANCE CURRICULUM (Atanassov 2025 local-difficulty): start the forward dx
                # range tiny (default 0 = pure in-place = the proven vertical-jump discovery)
                # and grow the upper bound ONLY after the policy STABLY LANDS ON TARGET at the
                # current distance (combined hit-AND-success rate, see _update_dx_curriculum).
                # Sidesteps the from-scratch discovery cliff: at dx=0 the landing target IS the
                # spawn, so the (tight-sigma) landing reward is fully available during discovery.
                self.landing_dx_max = float(getattr(self.cfg.commands, "landing_dx_start", 0.0))
                self.command_ranges["lin_vel_x"] = [0.0, self.landing_dx_max]
                self._far_stable_sum = 0.0   # (global advance-only accumulators — inert under per-env curriculum)
                self._far_n_sum = 0.0
                self._dx_last_advance_step = 0
                # PER-ENV 双向课程 (2026-07-04): 每个 env 自己的 dx 上界 (初始 landing_dx_start). 命令从
                # [0, 自己上界] 抽; 落地 hit→+step_up / miss→−step_down. landing_dx_max 每步 = 全局最大(仅日志).
                self.landing_dx_env = torch.full(
                    (self.num_envs,), float(getattr(self.cfg.commands, "landing_dx_start", 0.0)), device=self.device
                )
                # frontier-probe 掩码: 本集命令探到上界之外的 env (排除出课程升降, 见 _update_jump_state).
                self._dx_probe_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                # 本集这一跳是否"被计入的挑战跳"(起跳时锁: far & 非probe). 在 episode 末(_log_jump_episode_stats)评估升降,
                # 命中(jump_target_hits>=1)→+step_up, 否则(落短/摔/够不到都算)→−step_down. 这样"摔/跳废"也算 down, 不再漏算.
                self._ep_counted_far = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            else:
                self.command_ranges["lin_vel_x"] = list(self.cfg.commands.landing_disp_x_stage2)

        # Per-episode accumulator: jumps that landed within tol of the commanded landing point
        # (mirrors successful_jumps; consumed ONLY by the distance-curriculum advance gate).
        self.jump_target_hits = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # Commanded displacement magnitude of the episode's jump (recorded at touchdown), used to
        # filter the FAR-BAND advance gate (only jumps near dx_max count).
        self._last_jump_cmd_dx = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # Takeoff-omega gate: once the policy can jump (succ_rate EMA >= threshold), LATCH on a stronger
        # base_ang_vel_xy that ALSO covers the push -> suppress the nose-down spin AT TAKEOFF (it can't be
        # undone in flight). Gated on succ_rate (not a fixed step) so it adapts to discovery speed and never
        # penalizes the messy exploratory pushes before the robot can jump (that broke discovery before).
        self._succ_rate_ema = 0.0
        self._takeoff_omega_on = False
        # HONEST far-band mastery: decaying accumulators for a TRUSTWORTHY smoothed far-band hit rate
        # (per-batch far_n ~0.1 -> landing_stable_hit_rate is pure noise). See _log_jump_episode_stats.
        self._farband_hit_acc = 0.0
        self._farband_n_acc = 0.0

    # ------------------------------------------------------------------ #
    # Reset — set landing target = spawn xy + commanded displacement.
    # The robot spawns facing +x (identity heading) so the displacement maps
    # directly to world xy. Stage 1: commands[0:2] == 0 -> target == spawn.
    # ------------------------------------------------------------------ #
    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)  # parent resamples commands[0:2] (the displacement) first
        init_x = float(self.cfg.init_state.pos[0])
        init_y = float(self.cfg.init_state.pos[1])
        self.landing_target[env_ids, 0] = self.env_origins[env_ids, 0] + init_x + self.commands[env_ids, 0]
        self.landing_target[env_ids, 1] = self.env_origins[env_ids, 1] + init_y + self.commands[env_ids, 1]
        self._update_dx_curriculum()

    def _resample_commands(self, env_ids):
        # BIASED command sampling (Atanassov local-difficulty): after the parent draws dx uniformly from
        # [0, dx_max], RE-DRAW a fraction (landing_dx_frontier_frac) of envs into the FAR frontier
        # [landing_dx_frontier_lo * dx_max, dx_max] so the policy practices mostly at the GOAL distance
        # (the farthest landing point), not spread thin over easy near commands. The remaining fraction
        # keeps the uniform draw -> retains the easy->hard gradient + near-distance skill. Runs INSIDE
        # super().reset_idx (BEFORE landing_target is computed from commands[0:2]), so the target matches.
        # Only DISTANCE is biased (height untouched -- the goal is a stable landing at the farthest point,
        # not the highest jump). In-place phase (dx_max == 0) is a no-op, so discovery is untouched.
        super()._resample_commands(env_ids)
        # PER-ENV 双向课程 (2026-07-04): 每个 env 从 [0, 自己上界 landing_dx_env] 抽 dx, 覆盖父类的 global 抽.
        # play/eval(landing_dx_curriculum=False) 走父类的 command_ranges[DX,DX], 不进这段.
        if (len(env_ids) > 0 and getattr(self, "landing_dx_curriculum", False)
                and getattr(self.cfg.commands, "landing_dx_percurr", False)
                and (not getattr(self.cfg.test, "use_test", False)) and hasattr(self, "landing_dx_env")):
            de = self.landing_dx_env[env_ids]
            n = len(env_ids)
            base = torch.rand(n, device=self.device) * de                       # 主体: [0, 自己上界]
            # FRONTIER PROBE: 一小撮 env 命令探到上界之外 [de, de*probe_hi] → 唤醒 forward_reach/takeoff_velocity_match
            # 的"跳更远"梯度(命令不超上界时休眠). 探测 EXPECTED 够不到, 由 _dx_probe_mask 排除出课程升降(不污染诚实).
            probe_frac = float(getattr(self.cfg.commands, "landing_dx_probe_frac", 0.0))
            probe_hi = float(getattr(self.cfg.commands, "landing_dx_probe_hi", 1.4))
            probe = torch.rand(n, device=self.device) < probe_frac
            probe_val = de * (1.0 + (probe_hi - 1.0) * torch.rand(n, device=self.device))
            self.commands[env_ids, 0] = torch.where(probe, probe_val, base)
            self._dx_probe_mask[env_ids] = probe
            return
        # 单跳: landing_target 只在 reset_idx 设为 spawn+dx (撤回连续跳的 per-resample root+dx 更新).
        if len(env_ids) == 0 or not getattr(self.cfg.commands, "landing_dx_biased", False):
            return
        dx_max = float(self.command_ranges["lin_vel_x"][1])
        if dx_max <= 1e-6:
            return
        frac = float(getattr(self.cfg.commands, "landing_dx_frontier_frac", 0.7))
        lo = float(getattr(self.cfg.commands, "landing_dx_frontier_lo", 0.8)) * dx_max
        pick = torch.rand(len(env_ids), device=self.device) < frac
        front_ids = env_ids[pick]
        if len(front_ids) > 0:
            self.commands[front_ids, 0] = lo + (dx_max - lo) * torch.rand(len(front_ids), device=self.device)

    # ------------------------------------------------------------------ #
    # Distance curriculum: jump-stat plumbing + advance logic.
    # ------------------------------------------------------------------ #
    def _reset_jump_buffers(self, env_ids):
        super()._reset_jump_buffers(env_ids)
        if hasattr(self, "jump_target_hits"):
            self.jump_target_hits[env_ids] = 0.0
        if hasattr(self, "_ep_counted_far"):
            self._ep_counted_far[env_ids] = False   # 清本集"计入挑战跳"标记 (升降在 _log 里已用完)

    def _start_jump(self, env_ids):
        # Seed the squat-bottom anchor at jump start (mirrors the parent resetting jump_min_base_z to
        # the current base_z): the deepest-load xy is then tracked DOWN from here during loading.
        super()._start_jump(env_ids)
        if len(env_ids) > 0:
            self.squat_root_xy[env_ids] = self.root_states[env_ids, :2]

    def _update_jump_state(self):
        # SQUAT-BOTTOM capture (BEFORE super updates jump_min_base_z): while loading, whenever the base
        # reaches a new low, record its xy -> squat_root_xy ends at this jump's deepest-squat point.
        _loading = self.jumping_state & (~self.has_taken_off)
        _new_low = _loading & (self.root_states[:, 2] < self.jump_min_base_z)
        if torch.any(_new_low):
            self.squat_root_xy[_new_low] = self.root_states[_new_low, :2]
        super()._update_jump_state()
        # ANTI-CHEAT squat-based target: at takeoff, LOCK landing_target = SQUAT-BOTTOM xy + commanded
        # (dx, dy). Landing accuracy is then the real distance squat-bottom -> land, NOT spawn -> land:
        # the "stand -> squat" ground creep (~0.42m observed) can no longer farm distance, while the
        # normal push-off travel (squat -> liftoff) still counts as part of the jump. Locked ONCE ->
        # a fixed world point through flight+landing. squat_root_xy holds this jump's deepest-load xy.
        if torch.any(self.just_took_off):
            self.landing_target[self.just_took_off, 0] = (
                self.squat_root_xy[self.just_took_off, 0] + self.commands[self.just_took_off, 0]
            )
            self.landing_target[self.just_took_off, 1] = (
                self.squat_root_xy[self.just_took_off, 1] + self.commands[self.just_took_off, 1]
            )
            # PER-ENV 课程: 起跳时锁"这一跳是否被计入的挑战跳"(命令 >= far_frac×自己上界 且 非 probe). episode 末
            # (_log_jump_episode_stats) 按 jump_target_hits 评估升降 → 摔/跳废/落短(hits=0)都算 down, 修漏算失败偏差.
            if (getattr(self, "landing_dx_curriculum", False)
                    and getattr(self.cfg.commands, "landing_dx_percurr", False)
                    and (not getattr(self.cfg.test, "use_test", False)) and hasattr(self, "landing_dx_env")):
                jto = self.just_took_off
                cmd_d = torch.norm(self.landing_target[jto, :2] - self.squat_root_xy[jto], dim=1)
                far_frac = float(getattr(self.cfg.commands, "landing_dx_per_env_far_frac", 0.6))
                self._ep_counted_far[jto] = (cmd_d >= far_frac * self.landing_dx_env[jto]) & (~self._dx_probe_mask[jto])
        # Accumulate "landed ON the commanded point" for the curriculum gate: a REAL jump
        # (peak gate, same as landing_position) whose recorded touchdown xy (self.landing_root_xy,
        # set by the parent at just_landed) is within tol of self.landing_target.
        if hasattr(self, "jump_target_hits") and torch.any(self.just_landed):
            tol = float(getattr(self.cfg.commands, "landing_dx_hit_tol", 0.12))
            min_peak = float(getattr(self.cfg.rewards, "landing_real_jump_min_peak", 0.40))
            err = torch.norm(self.landing_root_xy - self.landing_target[:, :2], dim=1)
            hit = self.just_landed & (self.peak_base_height >= min_peak) & (err <= tol)
            self.jump_target_hits += hit.float()
            # remember this jump's commanded distance (for the far-band advance gate)
            self._last_jump_cmd_dx[self.just_landed] = torch.norm(self.commands[self.just_landed, 0:2], dim=1)
            # (PER-ENV 升降已移到 episode 末 _log_jump_episode_stats: 那里能把"摔/跳废/落短"都算 down, 不像这里
            #  的 just_landed 只看"落了地的跳"→漏算失败→dx_env 虚涨. jump_target_hits 上面已累积供末尾评估.)

    def _log_jump_episode_stats(self, env_ids):
        super()._log_jump_episode_stats(env_ids)
        # PER-ENV 双向课程升降 (2026-07-04, episode 末评估, 修漏算失败): 对本集"被计入的挑战跳"(_ep_counted_far,
        # 起跳时锁的 far&非probe), 按 jump_target_hits(本集命中数, >=1=命中)升降 — 命中→+step_up, 否则(落短/摔/
        # 跳废, hits=0)→−step_down. 之前在 just_landed 评估会漏掉"没干净落地的失败"(摔了不进 just_landed)→dx_env 虚涨;
        # 移到 episode 末, 每个起跳的挑战跳都会被评一次(hits=0 就是 down). 在 _reset_jump_buffers 清缓冲之前跑.
        if (getattr(self, "landing_dx_curriculum", False)
                and getattr(self.cfg.commands, "landing_dx_percurr", False)
                and (not getattr(self.cfg.test, "use_test", False)) and hasattr(self, "landing_dx_env")):
            counted = self._ep_counted_far[env_ids]
            # STABLE-HIT gate (2026-07-05): 升档要求"又准又稳"—— 落点近(jump_target_hits) AND 站稳(successful_jumps).
            # 修复病根: 之前只看 jump_target_hits(落点近就升), 策略靠"往前扑落点凑近但摔了"也能顶高 dx_env
            # -> dx 无限虚涨到 1.16 而 fb_smooth 崩到 0.16、succ 掉到 0.61(慢性退化). 现在摔了(successful=0)就不升
            # -> dx 自停在能"又准又稳"命中的真实距离(~0.75), succ 不再被超调的猛扑拖垮.
            hit = (self.jump_target_hits[env_ids] >= 1.0) & (self.successful_jumps[env_ids] >= 1.0)
            up = env_ids[counted & hit]
            down = env_ids[counted & (~hit)]
            self.landing_dx_env[up] += float(getattr(self.cfg.commands, "landing_dx_step_up", 0.02))
            self.landing_dx_env[down] -= float(getattr(self.cfg.commands, "landing_dx_step_down", 0.10))
            self.landing_dx_env.clamp_(
                float(getattr(self.cfg.commands, "landing_dx_floor", 0.0)),
                float(getattr(self.cfg.commands, "landing_dx_final", 2.0)),
            )
            self.landing_dx_max = float(self.landing_dx_env.max().item())
        # Smooth the successful-jump rate and LATCH the takeoff-omega gate once it clears the threshold
        # (one-way: stays on, never flickers off on a noisy dip). Discovery-safe: succ ~0 until the robot
        # can jump, so the gate only opens post-discovery regardless of how long discovery took.
        if "successful_jump_rate" in self.extras.get("episode", {}):
            self._succ_rate_ema = 0.99 * self._succ_rate_ema + 0.01 * float(self.extras["episode"]["successful_jump_rate"])
            if self._succ_rate_ema >= float(getattr(self.cfg.rewards, "takeoff_omega_succ_gate", 0.80)):
                self._takeoff_omega_on = True
        jump_den = torch.clamp(self.jump_starts[env_ids], min=1.0)
        self.extras["episode"]["landing_hit_rate"] = torch.mean(self.jump_target_hits[env_ids] / jump_den)
        # COMBINED curriculum gate metric: a jump counts ONLY if the SAME jump BOTH landed on target
        # (jump_target_hits) AND ended in a stable/successful landing (successful_jumps). Single-jump
        # episodes -> each count is 0/1, so the AND of (>=1) is "this episode's jump did both". This
        # closes the loophole where "some jumps hit then topple + others land short but stable" cleared
        # the old separate hit/succ averages without any jump being both.
        stable_hit = (self.successful_jumps[env_ids] >= 1.0) & (self.jump_target_hits[env_ids] >= 1.0)
        self.extras["episode"]["landing_stable_hit_uniform"] = torch.mean(stable_hit.float() / jump_den)  # diagnostic (all dx)
        # FAR-BAND gate metric: only jumps whose commanded dx was in the top band [dx_max*(1-frac),
        # dx_max] count, so the curriculum advances only when the NEWEST/farthest distances are
        # stably hit -- not when the easy near commands carry a uniform average. (n pushed so the
        # gate can skip batches with no far-band jumps instead of biasing the EMA to 0.)
        far_frac = float(getattr(self.cfg.commands, "landing_dx_far_frac", 0.40))
        jumped = self.jump_starts[env_ids] >= 1.0
        far = jumped & (self._last_jump_cmd_dx[env_ids] >= self.landing_dx_max * (1.0 - far_frac))
        far_n = far.float().sum()
        far_hit = (stable_hit & far).float().sum()
        self.extras["episode"]["landing_farband_n"] = far_n
        self.extras["episode"]["landing_farband_hit"] = far_hit
        self.extras["episode"]["landing_stable_hit_rate"] = far_hit / torch.clamp(far_n, min=1.0)  # per-batch (noisy) — diagnostic only
        # HONEST smoothed far-band mastery: per-batch far_n ~0.1 -> the rate above is pure noise (0.0..0.35).
        # Accumulate hits/attempts with a slow decay so the RATE is trustworthy (zero-far-band batches just
        # decay both -> don't corrupt it). THIS is the number to WATCH for "can it hit the far end" — NOT
        # landing_hit_rate (uniform, near-command-inflated) NOR landing_dx_max (strongest single env).
        # Deterministic ground truth = scripts/eval_reach_ceiling.py (model_3000: ~0.6 total, wall at 0.6->0.7).
        self._farband_hit_acc = 0.995 * self._farband_hit_acc + float(far_hit.item())
        self._farband_n_acc = 0.995 * self._farband_n_acc + float(far_n.item())
        self.extras["episode"]["landing_farband_hit_smooth"] = torch.tensor(
            self._farband_hit_acc / max(self._farband_n_acc, 1.0), dtype=torch.float, device=self.device)
        # REAL curriculum state (curriculum_pd_prior is unreliable): general_scale ramps 0->1 (warmup->x0),
        # pd_alpha = 0.5*(1-general_scale). Watch this to SEE when PD actually fades (target: 1.0 by ~iter1200).
        self.extras["episode"]["curriculum_general_scale"] = torch.tensor(float(self.general_scale), device=self.device)

    def _update_dx_curriculum(self):
        # Advance the forward dx range ONLY once the policy has TRULY mastered the far end of the
        # current range -- i.e. a CUMULATIVE far-band stable-hit rate (same jump lands on target AND
        # lands stably), over MANY samples, AFTER an adaptation window. The earlier per-batch EMA was
        # fooled by tiny far-band batches (rate 0/1) spiking to threshold -> it advanced on NOISE,
        # not mastery (pushed dx_max to 1.6 with only ~0.5 real far-band rate, then training on the
        # unreachable far commands collapsed the policy back to in-place). This robust gate:
        #   (a) skips the first min_hold steps after an advance (policy still adapting -> don't count),
        #   (b) then ACCUMULATES far-band hits/attempts,
        #   (c) advances only when >= min_far_samples have accumulated AND the cumulative rate >= thr,
        #   (d) resets the accumulators on advance.
        # -> the curriculum self-limits at the distance the policy can SUSTAINABLY hit; it cannot
        #    over-advance past the achievable ceiling. Inert outside training / when disabled.
        if not getattr(self, "landing_dx_curriculum", False):
            return
        if getattr(self.cfg.test, "use_test", False):
            return
        # PER-ENV 双向课程 (2026-07-04): 升降在 _update_jump_state(just_landed) per-env 做; 这里只出日志,
        # 不做 global advance (取代下面 advance-only 那套 —— 它会被 PD辅助期+noise 冲虚高、只升不退).
        if getattr(self.cfg.commands, "landing_dx_percurr", False) and hasattr(self, "landing_dx_env"):
            self.landing_dx_max = float(self.landing_dx_env.max().item())
            ep = self.extras.setdefault("episode", {})
            # 2026-07-11: report the 90th-PERCENTILE bound as `landing_dx_max`, NOT the single-luckiest-env max.
            # The max over 4096 noisy per-env random walks is always a high outlier that does NOT match
            # deterministic play; p90 = the top-decile STABLY-reached distance (a far more honest "reach upper
            # bound"). The true max is kept as landing_dx_maxenv for reference.
            ep["landing_dx_max"] = torch.quantile(self.landing_dx_env.float(), 0.90)
            ep["landing_dx_maxenv"] = torch.tensor(self.landing_dx_max, device=self.device)
            ep["landing_dx_mean"] = self.landing_dx_env.mean()
            ep["landing_dx_min"] = self.landing_dx_env.min()
            return
        dx_final = float(getattr(self.cfg.commands, "landing_dx_final", 0.40))
        episode = self.extras.get("episode", {})
        fn = episode.get("landing_farband_n", None)
        fh = episode.get("landing_farband_hit", None)
        min_hold = int(getattr(self.cfg.commands, "landing_dx_min_hold_steps", 1500))
        adapting = (self.common_step_counter - self._dx_last_advance_step) < min_hold
        if (not adapting) and fn is not None and fh is not None:
            self._far_n_sum += self._to_float(fn)
            self._far_stable_sum += self._to_float(fh)
        cum_rate = self._far_stable_sum / max(self._far_n_sum, 1.0)
        # Surface the curriculum state to the training log.
        self.extras["episode"]["landing_dx_max"] = torch.tensor(self.landing_dx_max, dtype=torch.float, device=self.device)
        self.extras["episode"]["landing_dx_stable_cum"] = torch.tensor(cum_rate, dtype=torch.float, device=self.device)
        self.extras["episode"]["landing_dx_farsamples"] = torch.tensor(self._far_n_sum, dtype=torch.float, device=self.device)
        step = float(getattr(self.cfg.commands, "landing_dx_step", 0.10))
        # Advance gate: enough SUSTAINED far-band samples AND cumulative rate cleared. (Jun23_01-23-30 baseline:
        # ADVANCE-ONLY -- no revert/back-off; the curriculum self-limits by NOT advancing when the far band isn't hit.)
        if self.landing_dx_max >= dx_final - 1e-6:
            return
        min_samples = float(getattr(self.cfg.commands, "landing_dx_min_far_samples", 150))
        thr = float(getattr(self.cfg.commands, "landing_dx_stable_hit_threshold", 0.70))
        if self._far_n_sum >= min_samples:
            # A full WINDOW of far-band jumps accumulated -> evaluate. Advance if it cleared the bar;
            # EITHER WAY reset the window so the rate tracks the MOST RECENT window, not the whole
            # stage history. (Bug it fixes: a pure cumulative-since-advance never resets at the FIRST
            # stage -- no advance has happened -- so the early pre-/mid-discovery failures permanently
            # drag the average below thr and the gate stays stuck at dx_max=0 forever even after the
            # policy masters in-place: observed farsamples~950k, cum 0.26 while per-batch hit 0.93.)
            if cum_rate >= thr:
                self.landing_dx_max = min(self.landing_dx_max + step, dx_final)
                self.command_ranges["lin_vel_x"] = [0.0, self.landing_dx_max]
                self._dx_last_advance_step = int(self.common_step_counter)  # re-adapt at the new distance
            self._far_stable_sum = 0.0
            self._far_n_sum = 0.0

    # ------------------------------------------------------------------ #
    # Pose-target override — during the post-touchdown landing buffer the parent leaves the
    # joint pd/reward target at q_ground (foot 0.30 = a TALL stance, base~0.34). With the PD
    # prior faded to 0 that tall pose never returns to the pre-jump idle pose (default_dof_pos,
    # base~0.31), so the cmd=0 stance is inconsistent pre/post jump -> topple + continuous-mode
    # drift. Pull the landing hold back to the canonical idle stand instead (this flows into both
    # the residual PD prior AND the default_pos reward, which read self.default_joint_pd_target).
    # ------------------------------------------------------------------ #
    def _update_default_joint_pd_target(self):
        super()._update_default_joint_pd_target()
        self.default_joint_pd_target[self.landing] = (
            self.default_dof_pos.expand(self.num_envs, -1)[self.landing]
        )

    # ------------------------------------------------------------------ #
    # Observations — identical to the parent layout except the velocity-command
    # slot (commands[:, :3] * commands_scale) is replaced by the yaw-frame
    # landing-point error  Ryaw^T (p* - p_base) = [fwd_err, lat_err, 0].
    # Only the yaw component of orientation is used so pitch/roll in flight do
    # not scramble the target direction. 69-dim layout + mirror parity preserved.
    # ------------------------------------------------------------------ #
    def compute_observations(self):
        foot_contact_obs = self._get_contact_state().float()
        motor_fatigue = self.motor_fatigue.detach()

        err_world = self.landing_target[:, :2] - self.root_states[:, :2]
        _, _, yaw = get_euler_xyz(self.base_quat)
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        err_fwd = cos_y * err_world[:, 0] + sin_y * err_world[:, 1]
        err_lat = -sin_y * err_world[:, 0] + cos_y * err_world[:, 1]
        # Landing error is always the REAL-TIME value (no obs-hold): the policy must always know its true
        # position relative to the target -- zeroing it post-touchdown lies about the position AND injects a
        # DISCONTINUITY at touchdown (err snaps to 0) that spikes the value-function loss / exploration std.
        # What stops the post-landing "chase" hop is the COMMAND, not hiding the error: after landing cmd4=0
        # (binary stand) and NO jump reward is collectable, so hopping toward the still-distant target earns
        # nothing and only costs -> a policy trained on the clean 0/1 command learns to HOLD despite the error.
        landing_err_obs = torch.stack((err_fwd, err_lat, torch.zeros_like(err_fwd)), dim=-1)

        height_obs = torch.cat(
            (
                self.root_states[:, 2:3] * 2.0,
                (self.commands[:, 3:4] - self.root_states[:, 2:3]) * 2.0,
            ),
            dim=-1,
        )
        obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                landing_err_obs,                    # <- replaces commands[:, :3] * commands_scale
                self.commands[:, 3:4] * 2.0,
                self.commands[:, 4:5],
                height_obs,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                foot_contact_obs,
                self.torques,
                motor_fatigue,
                self.pd_prior_alpha,
            ),
            dim=-1,
        )
        obs_buf = torch.nan_to_num(obs_buf, nan=0.0, posinf=100.0, neginf=-100.0)
        if self.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec
        self.obs_buf = torch.where(
            torch.rand(self.num_envs, device=self.device).unsqueeze(1) > self.cfg.domain_rand.loss_rate,
            obs_buf,
            self.obs_buf,
        )

        if self.num_privileged_obs is not None:
            feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
            feet_pos_local = feet_pos - self.root_states[:, :3].unsqueeze(1)
            feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]
            feet_contact_forces = self.contact_forces[:, self.feet_indices, :]
            self.privileged_obs_buf = torch.cat(
                (
                    obs_buf,
                    self.root_states[:, 2:3],
                    self.base_lin_vel,
                    feet_pos_local.reshape(self.num_envs, -1),
                    feet_vel.reshape(self.num_envs, -1),
                    feet_contact_forces.reshape(self.num_envs, -1),
                ),
                dim=-1,
            )

    # ------------------------------------------------------------------ #
    # Landing rewards (new). The jump-driving stack (takeoff_vz, projected_peak,
    # successful_jump, ...) is inherited from the parent untouched.
    # ------------------------------------------------------------------ #
    def _jump_commanded(self):
        if self.cfg.commands.num_commands > 4:
            return self.commands[:, 4] > float(self.cfg.commands.jump_command_threshold)
        return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def _landing_kernel(self, err_sq, base_sigma_key, norm_sigma_key):
        # exp kernel on a squared landing error, with optional DISTANCE-NORMALIZATION (Yang 2023):
        # divide err by the commanded displacement^2 so the reward is SCALE-INVARIANT -- a 15% miss
        # at 1.2m is judged like a 15% miss at 0.4m. A FIXED-sigma absolute kernel instead vanishes
        # at far targets (exp(-large) ~ 0), which is exactly why the policy plateaus at a ~constant
        # RELATIVE undershoot (lands at ~85% of the command). Floor avoids blow-up for in-place cmd.
        if bool(getattr(self.cfg.rewards, "landing_err_normalize", False)):
            floor = float(getattr(self.cfg.rewards, "landing_norm_dist_floor", 0.30))
            denom = torch.clamp(torch.norm(self.commands[:, 0:2], dim=1), min=floor) ** 2
            err_sq = err_sq / denom
            sigma = float(getattr(self.cfg.rewards, norm_sigma_key, 0.025))
        else:
            sigma = float(getattr(self.cfg.rewards, base_sigma_key, 0.10))
        return torch.exp(-err_sq / max(sigma, 1e-4))

    def _reward_projected_landing(self):
        # Olsen 2025 densification: while airborne, project the final landing xy
        # from ballistic motion and reward closeness to the commanded landing
        # point. Converts the otherwise terminal-only landing signal into a dense
        # in-flight gradient.
        #   t_land solves  pz + vz·t - ½g·t² = h_land  (later/descending root).
        #
        # HEIGHT GATE (pz > min): the `airborne` flag alone (feet off ground) is
        # NOT enough — it is farmable. A legs-tucked sprawl (feet off the ground,
        # body collapsed to ~0.13m over the spawn point) keeps `airborne` True, so
        # the projected landing trivially ≈ target and this dense reward pays
        # ~1/step forever without any real jump (observed: rew exploded 0.2→5.5,
        # mean_peak collapsed to 0.13). Requiring the body above standing height
        # mirrors the real-jump peak gate on the sparse landing_position term:
        # a genuine jump cannot hover up there, so the farm is impossible while
        # the dense in-place landing control (Stage 1, target=spawn) is preserved.
        pz = self.root_states[:, 2]
        min_h = float(getattr(self.cfg.rewards, "projected_landing_min_height", 0.40))
        active = (self.airborne & (pz > min_h) & self._jump_commanded() & self._squat_deep_enough()).float()
        g = 9.81
        vz = self.root_states[:, 9]
        h_land = self.env_origins[:, 2] + float(self.cfg.rewards.base_height_target)
        disc = torch.clamp(vz * vz + 2.0 * g * (pz - h_land), min=0.0)
        t_land = (vz + torch.sqrt(disc)) / g
        proj_xy = self.root_states[:, :2] + self.root_states[:, 7:9] * t_land.unsqueeze(1)
        err = torch.sum(torch.square(proj_xy - self.landing_target[:, :2]), dim=1)
        reward = self._landing_kernel(err, "sigma_landing_proj", "sigma_landing_proj_norm")  # exp: precision near
        # NON-VANISHING far PULL: the exp kernel ~0 for a big undershoot at a FAR target -> no gradient -> the
        # policy never learns to launch far enough and the curriculum plateaus where the gradient dies (~1.3).
        # Add a LINEAR term on the actual projected miss: clamp(1 - dist/d_ref) gives PARTIAL CREDIT + a CONSTANT
        # gradient pulling the ballistic projection toward the target at ANY distance (e.g. undershoot 1.3m by
        # 0.3m -> 0.8, still a slope to reduce it). exp keeps doing precision near; linear does "reach to it" far.
        if bool(getattr(self.cfg.rewards, "landing_lin_pull", False)):
            dist = torch.sqrt(err + 1e-8)
            d_ref = float(getattr(self.cfg.rewards, "landing_lin_ref", 1.5))
            lin = torch.clamp(1.0 - dist / max(d_ref, 1e-3), min=0.0)
            reward = reward + float(getattr(self.cfg.rewards, "landing_lin_coef", 0.5)) * lin
        return active * reward

    def _reward_takeoff_velocity_match(self):
        # MERGED launch reward (REPLACES takeoff_vertical_velocity): reward the takeoff velocity VECTOR matching
        # the ballistic launch needed to land on the commanded point AT the commanded apex height -> drives jump
        # HEIGHT (vz) and DISTANCE (forward v) together, as ONE reward. Fills the gap the closeness rewards
        # cannot: landing-accuracy already pays most of its value at an undershoot (diminishing returns), so
        # nothing CAUSE-side pushed the launch FAR enough. Vector-MATCH (not a direction dot-product) so a too-
        # vertical over-launch can't fake the forward requirement; LINEAR (not exp) so there's a gradient from
        # zero velocity (discovery). At dx=0 the required vector is purely vertical -> reduces to
        # takeoff_vertical_velocity (discovery-safe) and subsumes horizontal_drift (sideways = mismatch).
        base_height = self.root_states[:, 2]
        min_height = float(getattr(self.cfg.rewards, "ascending_min_base_height", 0.18))
        vz = self.root_states[:, 9]
        ascending = self.jumping_state & (vz > 0) & (~self.has_landed) & (base_height > min_height)
        ascending = ascending & self._squat_deep_enough()           # same squat-depth gate as takeoff_vz
        # required LAUNCH velocity (world frame): vz from the height cmd (== takeoff_vz target); horizontal =
        # commanded displacement / ballistic flight time (symmetric arc, land ~ launch height -> T = 2 vz/g).
        h_stand = float(getattr(self.cfg.rewards, "stance_standing_height", 0.30))
        vz_req = torch.sqrt((2.0 * 9.81 * (self.commands[:, 3] - h_stand)).clamp(min=0.01)).clamp(min=0.5)
        horiz_disp = self.landing_target[:, :2] - self.takeoff_root_xy   # world-frame intended displacement
        d = torch.norm(horiz_disp, dim=1)
        flight_t = (2.0 * vz_req / 9.81).clamp(min=1e-3)
        dir_xy = horiz_disp / d.clamp(min=1e-6).unsqueeze(1)
        v_req = torch.cat([(d / flight_t).unsqueeze(1) * dir_xy, vz_req.unsqueeze(1)], dim=1)   # (N,3)
        v_act = torch.cat([self.root_states[:, 7:9], vz.unsqueeze(1)], dim=1)                   # (N,3) world vel
        match = torch.clamp(
            1.0 - torch.norm(v_act - v_req, dim=1) / torch.norm(v_req, dim=1).clamp(min=1e-3),
            min=0.0, max=1.0,
        )
        return ascending.float() * match

    def _reward_launch_pitch_toward_vel(self):
        # 起跳上升段，奖励"机身鼻尖方向"对齐"起跳速度方向"(抬头往速度方向)。
        # 逼后腿驱动的抬头起跳弧 -> 召后髋/后大腿 -> 更多冲量。任务空间姿态，不被蹲浅绕过。
        base_h = self.root_states[:, 2]
        min_h  = float(getattr(self.cfg.rewards, "ascending_min_base_height", 0.18))
        vz = self.root_states[:, 9]
        ascending = self.jumping_state & (vz > 0) & (~self.has_landed) & (base_h > min_h) & self._squat_deep_enough()
        v = self.root_states[:, 7:10]
        vdir = v / torch.norm(v, dim=1, keepdim=True).clamp(min=0.3)      # 速度方向(上-前)
        fwd  = quat_apply(self.base_quat, self.forward_vec)               # 机身鼻尖方向(世界系)
        align = (fwd * vdir).sum(dim=1)                                   # cos夹角 ∈[-1,1];机身水平时≈cos(50°)≈0.64
        return ascending.float() * torch.clamp(align, min=0.0)           # 鼻尖越指向速度越高，自然封顶在"完全对齐"

    def _reward_default_hip_pos(self):
        # COMMAND-CONDITIONAL hip-abduction lock (2026-07-11, lateral unlock). The parent term rewards keeping the four
        # hip-abduction joints near default, which stops a forward jump from sliding the feet inward -- but it also forbids
        # the hip abduction a SIDE jump needs. So we keep the full lock for forward commands (d_y=0) and relax it toward 0
        # as the commanded lateral displacement grows: forward jumps stay anti-slide, lateral jumps are free to abduct.
        hip_ids = [0, 3, 6, 9]
        hip_error = torch.sum(torch.abs(self.dof_pos[:, hip_ids] - self.default_dof_pos[:, hip_ids]), dim=1)
        r = torch.exp(-self.cfg.rewards.default_hip_pos_gain * hip_error)
        dy_ref = float(getattr(self.cfg.rewards, "default_hip_pos_lat_ref", 0.15))
        lat_free = torch.clamp(self.commands[:, 1].abs() / dy_ref, 0.0, 1.0)   # 0 forward -> 1 fully lateral
        return r * (1.0 - lat_free)

    def _reward_forward_reach(self):
        # DISTANCE-PROGRESSIVE EFFORT reward, DECOUPLED from precise landing. Diagnosis: every jump reward
        # (projected_landing/landing_position/takeoff_velocity_match/successful_jump) is tied to HITTING the exact
        # point; once a far command is hard to hit precisely, all of them go hit-or-miss -> the policy loses a
        # stable positive signal and ABANDONS the big jump (squatQ collapses) -> the chronic post-peak degradation.
        # This rewards the projected FORWARD reach ALONG the commanded direction, by ABSOLUTE metres, capped at the
        # command (no overshoot bonus): jumping FARTHER pays MORE, and a FAR command pays more than a near one ->
        # the policy is ALWAYS rewarded for trying its best / reaching farther, even when it can't hit precisely,
        # so it never gives up. Precise landing stays a separate BONUS. Same in-flight ballistics + height gate as
        # projected_landing (farm-proof: a legs-tucked sprawl can't clear the height gate).
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
        reach_along = (reach_vec * cmd_dir).sum(dim=1)                  # signed reach along the command direction
        return active * torch.minimum(reach_along.clamp(min=0.0), cmd_dist)

    def _reward_four_leg_push(self):
        # WAKE THE IDLE LEGS WITHOUT forcing front==rear. torque_diag: the push is FRONT-loaded (front thighs
        # ~0.7-0.93 throttle) while the REAR thighs idle at ~0.3 -> wasted impulse -> shorter jump. But a forward
        # jump is biomechanically REAR-DRIVEN and the takeoff is STAGGERED (front lifts first), so we must NOT
        # force the four legs to push equally/together. Instead: each ON-GROUND leg is rewarded for pushing up to
        # `target` vertical GRF (saturates at target -> surplus is FREE = no front==rear constraint, rear may
        # dominate); a leg that has already LIFTED OFF (front-first takeoff) is EXCLUDED from the mean, not
        # penalized. An idle leg that is STILL on the ground (low GRF) drags the mean down -> the policy must push
        # it -> extracts the idle rear-thigh capacity (the hip is NOT velocity-limited, unlike the calf). Total
        # push MAGNITUDE stays driven by projected_peak / takeoff_velocity_match; THIS only stops legs idling.
        # DISCOVERY-SAFE GATES (push-off has crashed discovery before): succ-latch (_takeoff_omega_on) + push
        # phase + a real squat + a MEANINGFUL total force (> floor, above body weight = the actual push, not the
        # static squat-load where legs merely bear weight).
        if not getattr(self, "_takeoff_omega_on", False):
            return torch.zeros(self.num_envs, device=self.device)
        contact = self._get_contact_state()
        fz = torch.clamp(self.contact_forces[:, self.feet_indices, 2], min=0.0)      # (N,4) vertical GRF per foot
        total = fz.sum(dim=1)
        floor = float(getattr(self.cfg.rewards, "four_leg_push_force_floor", 200.0))  # > body weight = a real push
        active = self.jumping_state & (~self.has_taken_off) & self._squat_deep_enough() & (total > floor)
        target = float(getattr(self.cfg.rewards, "four_leg_push_force_target", 90.0))  # per-leg "contributing" force
        per_leg = torch.clamp(fz / target, 0.0, 1.0)                                  # each leg saturates at target
        in_contact = contact.float()
        graded = (per_leg * in_contact).sum(dim=1) / in_contact.sum(dim=1).clamp(min=1.0)  # mean over ON-GROUND legs
        return active.float() * graded

    def _reward_landing_position(self):
        # DENSE over the landing/settling phase (Atanassov 2025 'base position landing'), using the
        # FIXED touchdown xy (self.landing_root_xy, set at just_landed) so it cannot be farmed by
        # crawling toward the target after a short landing. Rewards "touched down on target" held
        # through the whole landing buffer -> revives this signal (the old SPARSE 1-step version
        # earned ~0 and was dead weight despite w30). Distance-normalized like projected_landing.
        # NOTE: dense -> ~150x the old per-jump magnitude, so its WEIGHT was cut hard in config.
        min_peak = float(getattr(self.cfg.rewards, "landing_real_jump_min_peak", 0.40))
        real_jump = self.peak_base_height >= min_peak
        active = self.landing.float() * real_jump.float()
        err = torch.sum(torch.square(self.landing_root_xy - self.landing_target[:, :2]), dim=1)
        return active * self._landing_kernel(err, "sigma_pos_landing", "sigma_pos_landing_norm")

    def _get_successful_jump_velocity_score(self):
        # DECOUPLED from the landing point (user, WITH clean_takeoff_terminate ON): successful_jump =
        # upright(binary) x height_score x THIS, and THIS returns ~1.0 for ANY clean stable jump regardless of
        # WHERE it lands. WHY: clean_takeoff_terminate forbids the re-plant, so if the bonus is COUPLED (paid only
        # when you land ON the far target), the ONLY way to earn it at an OVER-REACH command is a momentum-building
        # STUTTER-STEP -- which clean_takeoff then TERMINATES -> the expected ~1000 collapses to 0 -> the TD-error
        # blew the critic (value_loss 1.96 @iter716, Jun23_01-23-30) and destroyed the policy. Decoupled, a CLEAN
        # jump that falls SHORT still earns the bonus -> the policy is content to land short CLEANLY instead of
        # stutter-stepping -> graceful PLATEAU at the clean-jump reach, no crash. Accuracy stays driven by
        # landing_position/projected_landing. success_landing_min_score=1.0 fully decouples; lower (<1) to re-fold.
        floor = float(getattr(self.cfg.rewards, "success_landing_min_score", 1.0))
        if floor >= 1.0:
            return torch.ones(self.num_envs, device=self.device)
        err = torch.sum(torch.square(self.landing_root_xy - self.landing_target[:, :2]), dim=1)
        score = self._landing_kernel(err, "sigma_pos_landing", "sigma_pos_landing_norm")
        return floor + (1.0 - floor) * score

    def _reward_default_pos(self):
        # PENALTY on L1 pose deviation, but HALVED in config (-1.0 -> -0.5). Diagnosis (Jun21 runs): at -1.0 this
        # was the DOMINANT term (-0.81/s = 57% of all penalties) and TAXED the jump (a jump must deviate from the
        # pose target) -> jumping went net-negative -> policy death-spiralled into NOT jumping (best ~iter500,
        # collapse iter700-1100). Halving the weight cuts the jump tax ~in half (-0.81 -> ~-0.4) so jumping stays
        # net-positive, while keeping it a PENALTY (simpler than a reward; avoids a standing-pose reward that would
        # make not-jumping comfortable). Still zeroed during the ground push-off extension (legs must extend beyond
        # q_ground to launch) and the squat-DOWN (mid-fold). NOTE: memory says -0.7 once caused noise runaway --
        # watch noise_std; raise back toward -0.7/-1.0 if the pose anchor gets too loose.
        # FULL PD / default_pos ALIGNMENT (2026-07-15, user): NO phase is zeroed. default_pos penalizes the
        # L1 deviation from the phase-dependent PD target q* in EVERY phase, so it EXACTLY mirrors the pose
        # the PD prior pulls toward (squat / push / air / preland / land / idle) -- one target for both the
        # scaffold and the reward, including the rear-extended q_ground backward-push launch pose.
        return torch.sum(torch.abs(self.dof_pos - self.default_joint_pd_target), dim=1)

    def _reward_grounded_jump(self):
        # MUST-LAUNCH penalty (landing override) -- CLOSE the "retreat to NOT jumping" escape. Diagnosis (Jun21
        # runs): when the dx curriculum pushed past the actuator's physical reach, far jumps could not hit ->
        # the jump went net-negative, so the policy DROPPED below the squat gate (shallow/no jump) where it earned
        # nothing BUT also dodged the jump-motion penalties (default_pos/aerial_dof_acc/...). That "safe" no-cost
        # retreat collapsed the whole (shared) policy. Goal of the task is to JUMP, so make NOT launching the worst
        # option: once a jump is commanded and the squat window (grounded_grace_steps) has passed, penalize STILL
        # being fully grounded (not taken off). The policy can no longer dither/not-commit -> it must TRY ITS BEST
        # to launch (and the squat-gated jump rewards still favour a proper squat-jump over a bare pop).
        # DISCOVERY-SAFE: gated on the succ-EMA latch (_takeoff_omega_on, same gate as clean_landing) so it never
        # forces premature pops before the policy has learned the squat-jump.
        if not getattr(self, "_takeoff_omega_on", False):
            return torch.zeros(self.num_envs, device=self.device)
        contact = self._get_contact_state()
        all_feet_contact = torch.all(contact, dim=1)
        grace_elapsed = self.jump_step_counter > self.cfg.rewards.grounded_grace_steps
        stuck = self.jumping_state & (~self.has_taken_off) & grace_elapsed & all_feet_contact
        return stuck.float() * (0.5 + self._get_height_progress())

    def _reward_foot_contact_sync(self):
        # LEFT-RIGHT contact-timing sync ONLY (was four-foot sync). The takeoff/landing should be
        # left-right symmetric, but FRONT-REAR may STAGGER: the front feet may LIFT FIRST and the rear
        # feet push off LAST (the natural animal long-jump "rolling" takeoff). The old four-foot version
        # penalized ANY 1-3-feet-on state, which forbade that stagger (front-pair-off / rear-pair-on
        # looks "mixed") and forced a flat simultaneous takeoff. Now penalize only a LEFT vs RIGHT
        # mismatch WITHIN the front pair (FL!=FR) and WITHIN the rear pair (RL!=RR) -- a front-off /
        # rear-on state is left-right clean -> NOT penalized. Feet order [FL,FR,RL,RR] (idx 0,1 front
        # pair; 2,3 rear pair), per _reward_left_right_contact_sync. Penalty form (negative weight) ->
        # 0 when both pairs are L-R synced. Active in the push + landing-buffer transition windows.
        contact = self._get_contact_state()
        lr_stagger = (contact[:, 0] != contact[:, 1]).float() + (contact[:, 2] != contact[:, 3]).float()
        active = (self.jumping_state & (~self.has_taken_off)) | self.landing
        return active.float() * 0.5 * lr_stagger        # [0,1]: 0 = both pairs L-R synced, 1 = both staggered

    def _reward_stance_squat(self):
        # Pose-guided countermovement (GUIDE to the target, don't block cheats). The earlier
        # height-based version rewarded "base_z low", which is degenerate -- a single scalar that
        # the policy farmed by face-planting forward (@3000) and splaying the legs sideways (@4900),
        # both of which drop base_z without a launchable squat. Whack-a-mole gating of each sprawl
        # never ends. Instead reward approaching the loaded squat POSE q_squat (a clean symmetric
        # vertical fold, neutral hips): a full-pose target has essentially ONE satisfying config,
        # so there is nothing degenerate to cheat -- only the clean squat scores, and base_z drops
        # as a consequence of the fold, not as the objective.
        #
        # Gate = jumping & ~has_taken_off & not-yet-in-pose (NO vz term -> no joint_angle_loaded
        # dead loop): drives stand -> squat from the moment the jump is commanded, then switches
        # off once the pose is reached so it never fights the leg extension during the push (the
        # squat-POSE gate _squat_deep_enough unlocks the jump chain from the same threshold).
        sigma = max(float(getattr(self.cfg.rewards, "squat_pose_sigma", 3.0)), 1e-4)
        reward = torch.exp(-self._squat_pose_err() / sigma)
        thr = float(getattr(self.cfg.rewards, "squat_pose_threshold", 0.0))
        # Keep paying the dip-shaping reward until the squat is QUALIFIED (held >= squat_hold_steps),
        # not merely touched: this is what gives the policy a reason to STAY folded through the hold
        # window instead of popping straight back up the instant pose_err first dips under thr.
        not_in_pose_yet = (~self.squat_qualified) if thr > 0.0 else torch.ones_like(self.jumping_state)
        active = self.jumping_state & (~self.has_taken_off) & not_in_pose_yet
        return active.float() * reward

    def _reward_base_ang_vel_xy(self):
        # Landing-stability lever borrowed from Olsen 2025 / Atanassov 2025 (papers): damp base
        # roll/pitch ANGULAR VELOCITY through flight AND landing so the body does not enter touchdown
        # already tumbling -- the root cause of the "lands then flips" failure. Returned as a positive
        # magnitude with a NEGATIVE scale (penalty), NOT a positive bell kernel: a kernel that only
        # fires during a short phase is cheap for the height/success drive to ignore (cf. joint_angle_*
        # sitting at ~0). It penalizes rotation RATE, not airborne time, so a CLEAN high jump (w~=0)
        # pays nothing -> it does not bias toward shorter/lower jumps. yaw (w_z) is excluded because it
        # may be commanded in Stage 2.
        # ONE-SIDED pitch rate (user): roll stays symmetric (any roll is bad), but the PITCH rate is penalized
        # only in the NOSE-DOWN direction (base_ang_vel[:,1] < 0 = nose-down spin). The nose-UP launch rotation
        # (rear-hip-driven) is FREE -> unlocks the pitch-gated rear-hip potential without imparting a forward
        # face-plant spin. (nose-down: projected_gravity[0]>0 AND base_ang_vel[1]<0, confirmed in torque_diag.)
        roll_sq = torch.square(self.base_ang_vel[:, 0])
        nose_down_rate = torch.square(torch.clamp(-self.base_ang_vel[:, 1], min=0.0))
        ang_vel_sq = roll_sq + nose_down_rate
        if getattr(self, "_takeoff_omega_on", False):
            # POST-DISCOVERY (succ_rate gate latched): ALSO penalize ω during the PUSH/extension (where the
            # nose-down spin is IMPARTED -- it can't be undone in flight) and apply a STRONGER weight, so the
            # policy launches WITHOUT the spin -> level flight -> flat landing (Atanassov: control ω, drive it
            # to 0 at landing). Excluded before the gate so it never blocks the messy from-scratch pushes.
            active = (self.phase_extended | self.airborne | self.prelanding | self.landing).float()
            return float(getattr(self.cfg.rewards, "takeoff_omega_gain", 4.0)) * active * ang_vel_sq
        active = (self.airborne | self.prelanding | self.landing).float()
        return active * ang_vel_sq

    def _reward_landing_impact(self):
        # Landing-stability lever borrowed from Olsen 2025 (papers: Ground force L2 / Soft impact):
        # penalize the hard vertical foot-force SPIKE during the landing window so touchdown is
        # cushioned instead of slammed. Bounded to [0,1] (saturates) and zero below ~standing weight
        # (landing_impact_force_floor) -> a soft floor, never an unbounded "wall" the policy flees.
        active = self.landing.float()
        fz = torch.sum(torch.clamp(self.contact_forces[:, self.feet_indices, 2], min=0.0), dim=1)
        floor = float(getattr(self.cfg.rewards, "landing_impact_force_floor", 150.0))
        norm = max(float(getattr(self.cfg.rewards, "landing_impact_force_norm", 1500.0)), 1e-3)
        excess = torch.clamp((fz - floor) / norm, min=0.0, max=1.0)
        return active * excess

    def _reward_pitch_level(self):
        # Strengthen PITCH attitude specifically. The base holds a persistent nose-down ("head-heavy")
        # pitch that the shared orientation term (-2.0, roll+pitch EQUALLY, hence weak on pitch alone)
        # does not flatten. projected_gravity[:,0] is the pitch tilt (0 when level, ~sin(theta) when
        # nose-down). Penalty (positive magnitude, negative scale) over the WHOLE jump cycle
        # (load -> push -> flight -> land) so the body is driven level throughout -- and since an
        # asymmetric front/rear push is what pitches the body, this also pressures a symmetric push.
        # Local to the landing task -> leaves the shared _reward_orientation untouched; stacks on top
        # of it so pitch ends up weighted more than roll. yaw/roll unaffected.
        # ONE-SIDED pitch (user): the launch may rotate NOSE-UP (抬头, the rear-hip-driven launch arc / long-jump
        # back-tilt) FREELY, but NOSE-DOWN (低头/前栽, the face-plant) is penalized. torque_diag showed the rear
        # HIP sits at ~27% utilization (73% headroom) while the body is already at tilt 0.30 (75% of the 0.40
        # fallover) -> the rear-hip potential is PITCH-gated. A symmetric square(pitch) forbade the nose-UP rotation
        # that a rear-driven launch needs, capping reach. So: penalize only nose-down; allow nose-up.
        pg = self.projected_gravity[:, 0]                       # >0 = nose-DOWN (前栽), <0 = nose-UP (抬头)
        nose_down = torch.square(torch.clamp(pg, min=0.0))      # one-sided: nose-up -> 0 (free)
        level = torch.square(pg)                                # symmetric (for a flat/level touchdown)
        # (1) whole jump cycle: forbid nose-DOWN, ALLOW nose-up.
        r = self.jumping_state.float() * nose_down
        # (2) LANDING: want LEVEL for a flat 4-foot touchdown -> SYMMETRIC (nose-up also bad at touchdown).
        land = (self.prelanding | self.landing).float()
        r = r + float(getattr(self.cfg.rewards, "landing_pitch_extra", 5.0)) * land * level
        # (3) post-discovery strong AIR leveling: AIRBORNE flight only -> one-sided (allow the nose-up launch arc;
        # the symmetric landing term (2) handles the flat touchdown). Gated on the succ-latch.
        if getattr(self, "_takeoff_omega_on", False):
            r = r + float(getattr(self.cfg.rewards, "jump_pitch_extra", 12.0)) * self.airborne.float() * nose_down
        return r
