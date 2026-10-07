# 远程训练手册（BayesDL 4090 服务器）

> 2026-09-14 配好。服务器：Ubuntu 22.04，2× RTX 4090 D 49GB，20 核，60GB 内存。

## 0. 必须记住的规则

- **先开 VPN** 才能连服务器（不开 VPN 端口不通，ssh 会卡住）。
- **训练一律放 tmux 里跑**，断线不会杀进程。
- **每开一个新 shell / tmux 窗口，先 `source /root/bayes-tmp/xiaoyulong/env.sh`**。
- **只动 `/root/bayes-tmp/xiaoyulong/` 下面的东西**。`/root/bayes-tmp` 是多人共用盘，大家都是 root，别的文件夹（包括公共的 `conda_pkgs`、`envs`）一律不碰；`xiaoyulong` 里原有的 `latent-safety-reproduction`、`WMP`、`yanbo_wmp`、`isaacgym` 也不要改。
- 服务器上用 **`python -m pip` / `python -m wandb`**，不要直接敲 `pip` / `wandb`（环境是从 yanbo_wmp 克隆的，这些命令的脚本头可能还指向 yanbo 的 python）。
- 服务器**连不上 GitHub**，代码通过本地 `git push bayes` 同步（见第 4 节）。
- **API key、密码不要写进 `/root/bayes-tmp`**（共用盘，所有人可读）。

## 1. 磁盘

| 路径 | 说明 |
|---|---|
| `/` （52GB） | 系统盘，**重启可能丢**（`/root/.netrc` wandb 登录就在这里） |
| `/root/bayes-tmp` | 缓存盘，**重启保留**；但**任务停止 4 天后自动清空** → 停任务前把要的 checkpoint 拷回本地 |
| `/root/bayes-gpfs-51dc3b823a984d2f8b0d2383f77ab2bf` | 网络盘，**不要用来训练**（慢） |
| `/root/bayes-pub` | 只读公共盘 |

## 2. 连接

本地终端（开着 VPN）：

```bash
ssh bayes
```

`~/.ssh/config` 里已配好 `bayes`（主机、端口 43745、密钥 `~/Desktop/student-tmp`、每 30s 心跳防掉线）。不用别名的完整写法：

```bash
ssh -i /home/qiwang/Desktop/student-tmp -p 43745 -o ServerAliveInterval=30 -o ServerAliveCountMax=10 root@5285cd3f777694df72061649d8af3c9d.bayesdl.com
```

- 登录时的 `setlocale: LC_ALL: cannot change locale` 警告无害，忽略。
- **VS Code**：装 Remote-SSH 插件 → `Remote-SSH: Connect to Host` → 选 `bayes` → 打开文件夹 `/root/bayes-tmp/xiaoyulong/Torque-jump`。选解释器 `/root/bayes-tmp/xiaoyulong/envs/sata/bin/python`。
  - 卡在 `Downloading VS Code Server`：本地 VS Code 设置里加 `"remote.SSH.localServerDownload": "always"`（本地下载后传上去）。
  - VS Code Server 装在 `/root/.vscode-server`（系统盘），服务器重启后会自动重装，正常。
  - **训练不要在 VS Code 自带终端里直接跑**（关窗口/断线进程会死），在里面 `tmux attach -t Torque` 再跑。
  - **代码只在一边改**：推荐本地改 → `git push bayes`。如果在服务器上改了，在服务器 commit，本地 `git pull bayes <分支名>` 拉回来，否则两边会冲突。

## 3. tmux

```bash
tmux ls                  # 看有哪些会话
tmux attach -t Torque    # 回到已有会话（最常用）
tmux new -s Torque       # 没有会话时新建
```

会话里：`Ctrl+b` 然后 `d` = 离开（进程继续跑）；`Ctrl+b c` = 新窗口；`Ctrl+b 0/1/2` 或 `Ctrl+b n` = 切窗口；`Ctrl+b [` = 翻看历史输出（`q` 退出）。

⚠️ 已经在 tmux 里时再 `tmux new -s Torque` 会报 `duplicate session`，这不是错误，说明会话已存在。

## 4. 服务器上的目录布局

```
/root/bayes-tmp/xiaoyulong/
├── env.sh                   # 每个 shell 先 source 它
├── envs/sata/               # 我们的 conda 环境（py3.8, torch 2.0.1+cu118, numpy 1.22.3）
├── isaacgym/                # isaacgym（已 pip -e 装进 sata）
├── Torque-jump.git/         # 裸仓库 = 本地 push 的目标，别在里面改东西
├── Torque-jump/             # 工作副本，训练在这里跑（legged_gym / rsl_rl 已 pip -e 装进 sata）
├── conda_pkgs/  pip_cache/  torch_extensions_sata/   # 我们自己的缓存
```

`env.sh` 做的事：激活 sata、`LD_LIBRARY_PATH=$CONDA_PREFIX/lib`（isaacgym 需要 libpython3.8）、`PYTHONNOUSERSITE=1`、缓存目录指到 xiaoyulong、`cd` 到 `Torque-jump/legged_gym`。
本地留有一份副本 `~/bayes_env.sh`。

## 5. 同步代码（本地改 → 服务器跑）

本地（开 VPN）：

```bash
cd ~/torque_jump2/SATA
git push bayes <分支名>        # 推到服务器；需要备份到 GitHub 时另外 git push origin <分支名>
```

服务器：

```bash
source /root/bayes-tmp/xiaoyulong/env.sh
cd /root/bayes-tmp/xiaoyulong/Torque-jump
git fetch
git checkout <分支名>
git pull
git log --oneline -1           # 核对和本地 commit 一致
```

目前推上去的分支：`history-nohead`（带历史，advance-only 课程）、`landing-track-rewards`（单帧 flat，advance-only 课程）。

## 6. wandb

已登录（在线模式可用，服务器能连 api.wandb.ai）。**服务器重启后 `/root/.netrc` 会丢，要重新登录**：

```bash
python -m wandb login --relogin
python -c "import wandb; print(wandb.Api().viewer.username)"   # 确认是自己的账号
```

如果哪天连不上 wandb：`export WANDB_MODE=offline`，数据存在 run 目录里，拷回本地后 `wandb sync`。

## 7. 训练

### 冒烟测试（新环境 / 服务器重启后先跑一次）

```bash
source /root/bayes-tmp/xiaoyulong/env.sh
CUDA_VISIBLE_DEVICES=0 python legged_gym/scripts/train.py --task=go2_omnijump_landing_torque --headless --num_envs=512 --max_iterations=5 --run_name=smoke
```

第一次会编译 gymtorch，一两分钟没输出是正常的。跑通后删掉测试目录（先列出来确认，再按精确路径删）：

```bash
ls -d logs/go2_omnijump_landing_torque/*_smoke
rm -rf logs/go2_omnijump_landing_torque/<上面列出的那个完整目录名>
```

### 正式训练（单卡）

```bash
source /root/bayes-tmp/xiaoyulong/env.sh
CUDA_VISIBLE_DEVICES=0 python legged_gym/scripts/train.py --task=go2_omnijump_landing_torque --headless --run_name=<名字>
```

- `CUDA_VISIBLE_DEVICES=0` 用第一张卡，`=1` 用第二张卡。
- 日志 / checkpoint 在 `legged_gym/logs/go2_omnijump_landing_torque/<日期>_<名字>/`，每 100 轮存一次 `model_N.pt`。
- 看 GPU：`nvidia-smi`。

### 两张卡同时跑两个不同分支（⚠️ 首次使用前要先验证）

`legged_gym` 是 `pip -e` 装的，`import legged_gym` 默认永远指向 `Torque-jump/`。要同时跑另一个分支，需要第二个工作副本，并用 `PYTHONPATH` 让它导入自己的代码：

```bash
cd /root/bayes-tmp/xiaoyulong
git clone -b landing-track-rewards Torque-jump.git Torque-jump-flat

# 新 tmux 窗口：
source /root/bayes-tmp/xiaoyulong/env.sh
cd /root/bayes-tmp/xiaoyulong/Torque-jump-flat/legged_gym
export PYTHONPATH=/root/bayes-tmp/xiaoyulong/Torque-jump-flat/legged_gym:/root/bayes-tmp/xiaoyulong/Torque-jump-flat/rsl_rl
python -c "import legged_gym, rsl_rl; print(legged_gym.__file__); print(rsl_rl.__file__)"
```

**两行路径都必须在 `Torque-jump-flat` 下面**才能开训；如果还指向 `Torque-jump`，说明 PYTHONPATH 没压过 editable 安装，先别跑，问 Claude。确认后：

```bash
CUDA_VISIBLE_DEVICES=1 python legged_gym/scripts/train.py --task=go2_omnijump_landing_torque --headless --run_name=<名字>
```

### 续训

```bash
SKIP_PD_FADE=1 CUDA_VISIBLE_DEVICES=0 python legged_gym/scripts/train.py --task=go2_omnijump_landing_torque --headless --resume --load_run=<run目录名> --checkpoint=<N> --max_iterations=<再加多少轮>
```

- **必须加 `SKIP_PD_FADE=1`**：runner 不恢复 env 的 step_count，不加会从头重新淡出 PD。
- `--max_iterations` 是**在 checkpoint 基础上再加**的轮数，不是总轮数。

## 8. 看训练 / 拷回结果

**tensorboard**（服务器端口不对外，用 ssh 转发）：

```bash
# 服务器（tmux 新窗口）
source /root/bayes-tmp/xiaoyulong/env.sh
python -m tensorboard.main --logdir logs/go2_omnijump_landing_torque --port 6006

# 本地（开 VPN）
ssh -N -L 6006:localhost:6006 bayes
# 浏览器打开 http://localhost:6006
```

**拷回本地**（给 Claude 分析 / play / 防 4 天清空）：

```bash
# 本地
scp -r bayes:/root/bayes-tmp/xiaoyulong/Torque-jump/legged_gym/logs/go2_omnijump_landing_torque/<run目录名> ~/torque_jump2/SATA/legged_gym/logs/go2_omnijump_landing_torque/
```

⚠️ play 时要用训练这个 run 的那个分支 / commit 的代码。

## 9. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `ssh bayes` 卡住没输出 | VPN 没开 |
| `Connection ... closed by remote host` | 断线了。重连 → `tmux attach -t Torque`，tmux 里的进程还在 |
| `libpython3.8.so: cannot open shared object file` / `Ninja is required` | 忘了 `source env.sh` |
| `PyTorch was imported before isaacgym modules` | 自己写的脚本里要先 `import isaacgym`（或 `legged_gym.envs`）再 `import torch` |
| 长命令粘贴后报 `usage:` 或 `No such file` | 命令被复制折成了两行，粘贴后确认是完整一行再回车 |
| 粘贴多行 `<<EOF` 卡在 `>` | 结束的 `EOF` 前面有空格。`Ctrl+C` 退出，改用 scp 传文件 |
| wandb 训练初始化卡住 | 服务器重启后没登录，见第 6 节 |
| `git clone https://github.com/...` 连不上 | 服务器连不上 GitHub，用第 5 节的 `bayes` 远程 |
