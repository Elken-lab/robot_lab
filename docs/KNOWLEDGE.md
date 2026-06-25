# 学习过程记录（URDF → 实机）

> 分支：`learn`  
> 练习机型：**Unitree H1**（`assets/my_h1.py`、`config/humanoid/unitree_my_h1/`）  
> **常用 task id：** Rough `RobotLab-Isaac-Velocity-Rough-Unitree-My_H1`；Flat **`RobotLab-Isaac-Velocity-Flat-Unitree-My_H1`**（plane 训+play 用 Flat，勿与 Rough ckpt 混用）  
> 详细命令与 H1 改 cfg 细则见 [LEARNING.md](./LEARNING.md)

按 **从 URDF 到部署实机** 的顺序分章；每章 **实践步骤** 写本机已做过的操作，**遇到的问题** 写当时踩坑与处理，**知识点** 写原理与排错。

---

## 总览

```text
第 1 章  环境与工具链
第 2 章  厂商 URDF 与描述包
第 3 章  Isaac asset 接入（my_h1.py）
第 4 章  Env 配置与 gym 注册
第 5 章  环境验证（zero_agent / random_agent）
第 6 章  RL 训练（train.py）
第 7 章  策略验证与导出（play.py → policy.pt）
第 8 章  Sim2Sim（rl_sar → MuJoCo）
第 9 章  实机部署（rl_sar → 真机）
第 10 章 实验记录（集中版，放在文档最后）

```

| 章 | 在哪做 | 当前进度（简要） |
| --- | --- | --- |
| 1 | 本机 + conda | 已能 `train` / `zero_agent` 拉起 Sim |
| 2 | `data/Robots/...` | URDF + meshes 在位，MuJoCo 看过 |
| 3 | `robot_lab/assets/` | `my_h1.py`：`hip_pitch=-0.28`、踝 `armature=0.01`、腿/腰 PD；spawn 可站一阵 |
| 4 | `config/.../unitree_my_h1/` | `list_envs` 可见 My_H1 |
| 5 | `scripts/tools/` | `zero_agent` / `random_agent` / `spawn_check` 已跑 |
| 6 | `train.py` | **My_H1 多轮 + Flat 线** + **robot_lab 官方 H1 baseline**（附录 C、D）+ **Isaac Lab 预训练对照**（附录 E） |
| 7 | `play.py` | 已支持 debug 字段验收；每轮 play 结论见第 10 章 |
| 8 | rl_sar → MuJoCo | **进行中**：policy/config/base/FSM/编译/首跑已做；**待** `dof_vel_scale`、base 落盘核对、MJCF sensor 复测、策略行走 |
| 9 | rl_sar → 实机 | 尚未进行 |

**分工牢记：** robot_lab = Isaac 里训；MuJoCo / Gazebo / 实机 = **rl_sar**。

---

## 第 1 章 环境与工具链

### 实践步骤

1. 使用 conda 环境 **`isaaclab`**（Isaac Sim 5.1 / Isaac Lab 2.3.2）。
2. 在 **`~/project/robot_lab`** 根目录跑脚本（`python scripts/...`）。
3. 本机曾跑通 **`train.py`**，终端有 iteration log。
4. 本机 **`zero_agent`** 能拉起 Sim。

### 知识点

| 工具 | 作用 | 和 ROS |
| --- | --- | --- |
| Isaac Sim / Isaac Lab | 训练用仿真 | 一般不用 |
| MuJoCo | 另一套物理仿真、viewer、rl_sar | 默认不用 ROS |
| Gazebo | 常配合 ROS launch | 绑得紧 |
| robot_lab | Isaac 里 RL 环境 | — |
| rl_sar | Sim2Sim / 实机跑策略 | 可选 ROS 桥 |

- 改 **已有** `.py` cfg：保存后 **重启脚本** 即可；`pip install -e` 仅在首次安装、改 `setup.py` 或新 task 注册不进去时再跑

---

## 第 2 章 厂商 URDF 与描述包

### 实践步骤

1. 使用仓库内已有目录：
   ```text
   source/robot_lab/data/Robots/unitree/h1_description/
   ├── urdf/h1.urdf
   ├── meshes/*.STL
   ├── package.xml
   └── mjcf/
   ```
2. 在 `h1_description/` 下用 MuJoCo 看过模型：
   ```bash
   cd .../h1_description
   python -m mujoco.viewer --mjcf=mjcf/scene.xml
   ```
   初始姿态偏「倒下」属正常，用于确认 mesh/关节树能加载。
3. 为第 3 章改 asset，对照 `h1.urdf` 里 **`type="revolute"`** 关节名（H1 约 19 个）。

### 遇到的问题

| 现象 | 原因 / 处理 |
| --- | --- |
| URDF 里 **`package://h1_description/meshes/...`** 会不会错？ | 厂商给 ROS 用的写法；mesh 在 `meshes/` 里就对。**接 Isaac 先不改**；Sim 报找不到 mesh 再改成 `../meshes/`。 |
| MuJoCo `mjcf/scene.xml` 里 **H1 是倒地的** | MJCF 默认姿态，不等于 Isaac 会倒；只用来确认能加载。 |
| 搞不清 **先写 package.xml 还是先写 URDF** | 顺序是 **URDF + meshes**，package.xml / mjcf 是后加的包装。 |

### 知识点

| 文件 | 作用 | Isaac 要不要 |
| --- | --- | --- |
| **URDF** | 关节树、限位、惯性、mesh 引用 | 要（经 asset 指向） |
| **meshes/** | 形状（STL 等） | 要 |
| **package.xml** | ROS 包名 + 依赖 | **不要** |
| **mjcf/** | MuJoCo 入口 | Isaac 不用 |

- **`imu_joint` 等 fixed**：不算 DOF，不进 actuators。
- **H1 vs G1**：H1 单踝 `*_ankle_joint`、腰 `torso_joint`；G1 双踝 + `waist_*` + 手腕。

---

## 第 3 章 Isaac asset 接入（`my_h1.py`）

### 实践步骤

1. 参考 `robot_lab/assets/unitree.py` 里 `UNITREE_G1_29DOF_CFG`，新建 **`robot_lab/assets/my_h1.py`**。
2. 已改：`asset_path` → `h1_description/urdf/h1.urdf`；导出 **`UNITREE_My_H1_CFG`**。
3. 已改 **`actuators`**：去掉 G1 独有（`waist_*`、双踝 `ankle_pitch/roll`、`wrist_*`），踝用 `.*_ankle_joint`，腰用 `torso_joint` 等 H1 名。
4. **`init_state.joint_pos`** 已按官方 H1 角填写（髋/膝/踝/肩/肘 + `torso_joint`）。
5. **`legs` 里 `torso_joint`** 已在 `effort_limit_sim` / `stiffness` / `damping` 等 dict 中单独配置；**`enabled_self_collisions=False`**（与官方 USD H1 一致）。
6. 本章用 **第 5 章 `zero_agent` / `spawn_check`** 验收 spawn；**zero 仍会倒**（官方 H1 task 的 zero 也会倒），不等于 asset 失败。

### 遇到的问题

| 现象 | 原因 / 处理 |
| --- | --- |
| `zero_agent` 崩：**`Not all regular expressions are matched`**，`ankle_pitch/roll`、`waist_*` 为 `[]` | `actuators` 仍写 G1 关节名。**处理：** 踝 `.*_ankle_joint`；删 `waist_*`、`wrist_*`；`torso_joint` 并进 legs 或单独一组。 |
| 改完 actuators 能 spawn，但 **仰躺** | 早期 `joint_pos` 仍是 G1。**处理：** 按官方 H1 改站立角（**已做**）。 |
| `torso_joint` 在 `legs` 组但 **无 PD 项** | 腰软塌 → 易倒。**处理：** 在各 motor dict 里为 `torso_joint` 补 effort/stiffness/damping 等。 |
| `enabled_self_collisions=True` | 自碰多 → 易扭倒。**处理：** 改为 `False` 对比 spawn。 |
| `torso_joint` 写在 `ImplicitActuatorCfg(...)` 外单独一行 | **SyntaxError**，包无法 import。**处理：** 只写在各 dict 内部。 |
| `Robot` **World Z≈1.05** 却躺着 | `pos` z 是骨盆 spawn 高度，不是站直；躺是 **joint_pos + 执行器 + 重力**。**处理：** 改角与 PD，不必盲目加 z。 |
| 踝 **`armature=100`**（G1 复制残留） | 踝锁死 → 先 upright 再 **后倒**。**处理：** 改为 **~0.01**。 |
| **`hip_pitch=-0.25`** | 侧倒。**处理：** 试 **-0.27～-0.28**（官方 **-0.28**）。 |

### 知识点

```text
h1.urdf           → 结构、关节名、限位（厂商）
my_h1.py          → 用哪个 URDF、spawn 高度、初始角、仿真电机 PD
env_cfg.py        → 任务、观测、奖励（下一章）
```

| `my_h1.py` 块 | 干什么 |
| --- | --- |
| `spawn` / `asset_path` | 导入哪个 URDF |
| `init_state.pos` | 根 link 生成高度（H1 ~1.05） |
| `init_state.joint_pos` | 生成关节角，正则要对 URDF 关节名 |
| `actuators` | 仿真电机分组；关节名必须 URDF 里存在 |

---

## 第 4 章 Env 配置与 gym 注册

### 实践步骤

1. 复制 **`config/humanoid/unitree_g1/`** → **`unitree_my_h1/`**。
2. **`rough_env_cfg.py`**：`from robot_lab.assets.my_h1 import UNITREE_My_H1_CFG`；`foot_link_name = ".*ankle_link"`。
3. **`flat_env_cfg.py`** / **`__init__.py`**：类名、task id 与 Rough 一致。
4. **`list_envs.py | grep -i my`** 能看到 **`RobotLab-Isaac-Velocity-Rough-Unitree-My_H1`**（及 Flat id）。

### 遇到的问题

| 现象 | 原因 / 处理 |
| --- | --- |
| import **`robot_lab.assets.unitree`** 里的 `UNITREE_My_H1_CFG` | cfg 在 **`my_h1.py`**。**处理：** 改为 `from robot_lab.assets.my_h1 import ...`。 |
| `flat_env_cfg` 仍是 **`UnitreeG1*`** | 复制 G1 后未改全。**处理：** `UnitreeMyH1RoughEnvCfg` / `UnitreeMyH1FlatEnvCfg`，`__init__.py` entry 一致。 |
| **`Overriding environment ... G1-v0`** | `__init__.py` 重复 register G1；警告可先忽略。 |

### 知识点

- **`foot_link_name`** 是 **link** 名，不是 joint。
- **Flat vs Rough**：你 zero 时用过 Rough，人落在台阶上仰躺。

---

## 第 5 章 环境验证（zero_agent / random_agent）

### 实践步骤

1. **`zero_agent`**：单环境、非 headless 打开 viewer，确认任务能 spawn、不报错；曾见 H1 **仰躺**（改 `joint_pos` 前）。
2. **`random_agent`**（`--num_envs 1`）：不崩；人仍躺地，随机动作下 **乱晃**（符合预期）。

### 遇到的问题

| 现象 | 原因 / 处理 |
| --- | --- |
| **`zero_agent`：Isaac 窗口能打开，界面卡住一会儿后进程崩掉** | **处理：** 在 `rough_env_cfg.py` 里注释与腰/torso 惩罚相关的两行。 |
| **视口里找不到人** | 默认 4096 env。**处理：** `--num_envs 1`；选 **`pelvis`** 按 **`F`**；Follow Mode → **Robot**。 |
| 人在 **地形一角仰躺** | 多 env 或 `joint_pos` 未对齐；Rough 上躺倒符合预期。 |
| 选 **`Robot` 按 F** 只对到 gizmo | 改选 **`pelvis`** 再 Frame。 |
| Property：**Center of Mass** 的 Z 不是坐标 | 看 **Transform → Translate**；**Offset**=Local，无 Offset≈World。 |

### 知识点

| | zero_agent | random_agent |
| --- | --- | --- |
| 动作 | 0 | 随机 |
| 验证 | spawn、配置、别立刻 NaN | 有动作时关节/接触别疯 |
| 进度 | 已做 | 已做 |

- **倒下不算失败**，说明还要改 `joint_pos`。
- 走得好不好是 **第 6 章 train** 的事。

---

## 第 6 章 RL 训练（train.py）

### 实践步骤

训练阶段只保留方法：确认任务名、训练日志目录、TensorBoard 曲线和 checkpoint 产物是否对应同一轮实验。

#### 训练记录

每轮实验的改动、曲线现象、play 验收和结论已统一移到文档最后的 **第 10 章：实验记录（集中版）**。本章只保留训练方法和 TensorBoard 读图方法。

### TensorBoard 怎么看（本任务分析顺序）

#### 通用读图

| 元素 | 含义 |
| --- | --- |
| **横轴 Step** | 训练 logger 记录的步数（**不是**走了多少米） |
| **纵轴 Value** | 多 env 上该指标的 **平均值** |
| **浅色线** | 原始值（抖） |
| **深色线（Smoothed）** | 看 **趋势** 用这个 |
| **悬停** | 读 Value / Smoothed / Step / 训练时长 |

**原则：** 看 **趋势与平台期**，不盯单点尖峰。

#### 各曲线含义与期望

| 搜索名 | 是什么 | 怎么看 | 好 / 坏（本任务） |
| --- | --- | --- | --- |
| **Train/mean_reward** | 所有奖励项 **加权和** 平均 | 涨后走平 = PPO 认为学不动了 | 高 **≠** 会走；可能只是局部最优 |
| **Train/mean_episode_length** | 一局 **活了多少仿真步**（上限 ~1000） | 接近 1000 = 少早死 | 顶满 **≠** 站着走；趴满 20 s 也是 1000 |
| **Episode_Termination/time_out** | 结束的局里 **超时** 的比例 | 接近 1 = 几乎都撑满时限 | 高 time_out 可能只是学会 **熬 timeout** |
| **Episode_Reward/track_lin_vel_xy_exp** | xy **速度跟踪**（乘 reward weight） | 同一 run 内越高越好；**高 ≠ 迈步** | 须与 **feet_air** 和 play 对照 |
| **Episode_Reward/feet_air_time** | **脚离地** 相关（鼓励迈步） | 应 **>0 且训练中长期上涨**；**>0.01** 更像在学走 | 仍须 play 验收是否交替迈步 |
| **Episode_Reward/undesired_contacts** | **非脚触地** 惩罚（weight<0 时越负触地越多） | 越接近 **0** 越好 | 只能说明少用身体蹭地，不能单独证明会走 |
| **Episode_Reward/upward** | **躯干竖直/朝上** | 较高说明姿态项不差 | 单独高不够；可与 feet_air_time **分开看** |
| **Episode_Termination/terrain_out_of_bounds** | 走出地形格 | 低 = 少出界 | 不能说明会走 |

#### 组合判读（口诀）

```text
mean_reward ↑ + ep_length ≈ 1000 + time_out ≈ 1
    → 「总指标收敛、活得久」（可能是躺赢）

feet_air_time ≈ 0 + track_lin_vel 涨但仍 ≪ 上限
    → 「跟踪数字变好、脚仍不着地」→ 可能是蹭地/滑/转圈，不是走路

feet_air_time 有上升 + track 也不错，但 play 单脚转圈
    → **能站的新捷径**；改命令/奖励开新 run，别堆 iter

feet_air 仍低 + undesired 改善，但 play 仍转圈
    → 关角速度命令不一定够；继续改奖励/约束

track 很高 + undesired 接近 0 + feet_air 下跌
    → **指标好看、仍可能不会走**；选 feet_air 峰值 ckpt play

play 躺地 + feet_air_time ≈ 0
    → 停训 / 改 Plan B，别堆 iter
```

#### 建议日常看板顺序

1. **feet_air_time** — 有没有在学走  
2. **track_lin_vel_xy_exp** — 有没有跟命令  
3. **Episode_Termination/**（time_out、illegal_contact 等）— 摔得多还是熬 timeout  
4. **mean_reward** — 总趋势  
5. **mean_episode_length** — 辅助  

**验收「会走」：** play 里 **双足跟命令** + 训练里 **feet_air_time、track_lin_vel 持续上升**；**不能**只看 mean_reward / time_out。

### 遇到的问题

| 现象 | 原因 / 处理 |
| --- | --- |
| **play 报 69 vs 66（`mlp.0.weight`）** | policy obs 与训时不一致（例如 `base_lin_vel` 关/开）；play 须对齐该 run 的 `env.yaml`。 |
| **play 报 `create_articulation_view` None** | `train.py` 仍在跑；先停训再 play。 |
| **feet_air 训练中期下跌** | 选峰值 ckpt play；不要盲目用末期 ckpt。 |
| **train 时无法同时 play** | 单 GPU 单 Isaac Sim；停训后再 play。 |
| **play `--checkpoint` 找不到文件** | 用完整 checkpoint 路径，或确认 `--load_run` 指向正确目录。 |
| **reward / time_out 很好但不会走** | 总 reward、episode length、track 高都不能代替 play 验收；以脚离地、接触交替、前进位移为准。 |

### 知识点

- **两种「收敛」：** 总 reward / track 收敛不等于走路收敛。
- **走路收敛最低要求：** `feet_air_time` 有明显抬升、play 不摔、持续前进、左右脚有交替离地/接触。
- **ckpt 选择：** 优先 play `feet_air` smoothed 峰值附近的 checkpoint；末期 ckpt 可能已经退化。
- **obs 对齐：** checkpoint actor 输入维度必须与训练时一致。
- **play 验收优先级：** `fallen/base_z/roll/pitch` → `vx/Δx` → 左右脚 `foot_z/contact` 是否交替。
- **每轮具体结论：** 见文档最后 **第 10 章：实验记录（集中版）**。

---

## 第 7 章 策略验证与导出（play.py）

### 实践步骤

1. 用训练 **同一个 task** 做 play，checkpoint 路径与 `experiment_name` 一致（Flat → `unitree_my_h1_flat`，Rough → `unitree_my_h1_rough`）。
2. **必带 `--num_envs 1`**（默认 64 机器人）；验收步态建议 **`--forward`**（固定 `vx=0.3`，避免随机零命令站桩）。
3. Flat task **不必** `--plane`；Rough task 验收 Rough 能力时 **不要** `--plane`。
4. `play.py` 输出 `fallen/base_z/roll/pitch/vx/Δx/left_contact/right_contact`；`cmd=[0,0,0]` 多为未开 `--forward` 或 `heading_command` 未关。
5. 每轮具体验收见 **第 10 章**。

### 遇到的问题

#### play 后倒地、一直躺（核心现象）

| 现象 | 说明 |
| --- | --- |
| **`play.py` 视口里机器人多呈倒地/匍匐**，单 env、多 env 均如此 | 与训练 task、地形格一致时仍 **看不出双足行走**；`--num_envs 1` 盯十几秒也 **常全程躺地**。 |
| **tensorboard 却显示** mean episode length **~943**、mean reward **~48**、`Episode_Termination/time_out` **~0.9**、`terrain_out_of_bounds` **≈0** | 曲线 **好看** 与 play **全倒** **可以同时成立**，不能互相推翻。 |

#### 原因分析（按优先级）

1. **策略局部最优**：站桩、蹭地、转圈、倒地滑行都可能拿到部分 reward。
2. **训练指标误导**：`track`、`time_out`、`mean_reward` 好看不等于会走。
3. **动作/PD/reward 不匹配**：动作尺度、关节 PD、脚离地奖励、接触惩罚需要一起验证，但每轮只改一项。
4. **play 与训练配置不一致**：obs 维度、checkpoint 路径、仍有 train 占 GPU 都会导致误判。

每轮具体现象和结论见文档最后 **第 10 章：实验记录（集中版）**。

#### 处理与验收（建议顺序）

| 步骤 | 做什么 |
| --- | --- |
| 1 | 确认训练进程已停，再 play。 |
| 2 | 确认 checkpoint、观测维度和当前配置一致。 |
| 3 | 用 debug 字段判断：先看是否摔倒，再看是否有持续位移，最后看左右脚是否交替。 |
| 4 | 每轮只根据一个实验结论决定下一项改动。 |

### 知识点

- **play** 验的是 **已训策略** 在仿真里的 **肉眼行为**；**tensorboard 的 Episode_Termination/** 验的是 **局怎么结束**，二者 **都不能单独代替「会走」**。
- **验收「会走」**：新 run 的 play 中 **双足明显跟踪速度命令** + 训练侧 **`track_lin_vel_xy` 等奖励上升**；不能只看 time_out 或 episode length。
- 导出 **`exported/policy.pt`** 在 `play` 跑完后位于该 run 的 `exported/` 下，供第 8 章 rl_sar。

#### Rough 地形：会不会「随机出现在不同坑里」

| 情况 | 说明 |
| --- | --- |
| **多 env（默认可能很多个）** | Rough 是一大块 **地形网格**，每格一种地貌（坑、台阶、尖刺、平地…）。**每个 env 的机器人 spawn 在各自那一格上**，所以镜头里会同时看到不同地貌；**不是**同一只 H1 在随机换坑。 |
| **单 env（`--num_envs 1`）** | 始终 **固定在这一格**；旁边别的坑/台阶是 **别的 env 的格子**（多 env 时才有），不是这一只跳过去的。 |
| **同一 env、每局 reset** | `randomize_reset_base` 只在 **当前格子内** 小范围随机 **x / y / 朝向**（约 ±0.5 m），**不会** 每次 reset 换到完全不同的坑格。 |
| **训练时** | 有 **terrain curriculum** 时，难度等级会变；和 play 里盯一两局不完全一样。 |

**一句话：** 地貌多样 = **多 env 各占一格**；单 env 不会随机刷到别的坑格，只在格内挪一点。

---

## 第 8 章 Sim2Sim（rl_sar → MuJoCo）

> **策略来源：** Flat-10 PASS · run `2026-06-15_13-15-36` · ckpt **`model_1000.pt`**（USD 训，勿用 4k+）  
> **目标：** Isaac 训好的 `policy.pt` → **rl_sar** → **MuJoCo** 里走起来（尚不上实机）  
> **仓库分工：** `robot_lab` 训+导出；`rl_sar` 加载策略+仿真；MJCF 在 `h1_description/mjcf/`

### 8.0 总流程（做到哪了）

```text
[✓] 1. MuJoCo 加载 H1 模型（厂商 MJCF，robot_lab 内 h1_description）
[✓] 2. Isaac play 导出 policy.pt（显式 model_1000.pt）
[✓] 3. rl_sar policy/h1/robot_lab/config.yaml + policy.pt
[✓] 4. rl_sar policy/h1/base.yaml（会话中已按步改完；须核对磁盘顶层 key 为 h1:）
[✓] 5. rl_sar_zoo/h1_description（MJCF + scene；h1.xml 已补关节 sensor）
[✓] 6. fsm_h1.hpp + fsm_all.hpp + ./build.sh -mj + rl_sim_mujoco h1 scene 首跑
[进行中] 7. 站起/RL 行走验收（sensor 补完后复测；dof_vel_scale→1.0）
```

#### 8.0.1 已改文件一览（路径均在 `~/project/rl_sar` 除非另注）

| 文件 | 作用 | 怎么改 / 现状 |
| --- | --- | --- |
| `policy/h1/robot_lab/policy.pt` | JIT 策略权重；`InitRL` 时 `torch::jit::load` | 从 robot_lab 导出目录 **cp**：`logs/rsl_rl/unitree_my_h1_flat/2026-06-15_13-15-36/exported/policy.pt`（**git 常忽略**，目录里可能 `ls` 看不到但运行时需要） |
| `policy/h1/robot_lab/config.yaml` | **策略配置**：69 维 obs 怎么拼、action_scale、PD、力矩、关节零位；FSM 进 RL 时读 `h1/robot_lab` 段 | 抄 `go2/robot_lab/config.yaml`，顶层 key 改为 **`h1/robot_lab:`**，维数与 Flat-10 对齐（见 §8.3） |
| `policy/h1/base.yaml` | **机器人底座**：`dt`/`decimation`、关节名、MuJoCo `joint_mapping`、站起用 `fixed_kp/kd`；`rl_sim_mujoco` 启动时 `ReadYaml("h1","base.yaml")` 读 **`h1:`** 段 | 抄 `go2/base.yaml`，顶层 **`h1:`**，12 维数组扩 **19**（见 §8.4） |
| `src/rl_sar_zoo/h1_description/` | MuJoCo 加载的 MJCF/mesh；`rl_sim_mujoco` 硬编码 `.../rl_sar_zoo/h1_description/mjcf/{scene}.xml` | 从 robot_lab 厂商包 **cp** `h1_description` 整目录；`scene.xml` include `h1.xml` |
| `src/rl_sar_zoo/h1_description/mjcf/h1.xml` | H1 刚体、actuator、**sensor** | `<sensor>` 由仅 IMU 改为 **19×jointpos + 19×jointvel + 19×jointactuatorfrc + framequat + gyro**（顺序同 `<actuator>`，见 §8.6） |
| `src/rl_sar/fsm_robot/fsm_h1.hpp` | H1 状态机：Passive → GetUp → RLLocomotion(GetDown) | **cp** `fsm_gr1t2.hpp` → 全局 `gr1t2`→`h1`；`RLFSMStateRLLocomotion` 里 `config_name = "robot_lab"` |
| `src/rl_sar/fsm_robot/fsm_all.hpp` | 编译期注册所有 FSM | 增加 `#include "fsm_h1.hpp"`（`REGISTER_FSM_FACTORY` 在 hpp 末尾） |
| `cmake_build/bin/rl_sim_mujoco` | MuJoCo 仿真可执行文件 | `cd rl_sar && ./build.sh -mj`（改 FSM 须重编；**只改 MJCF 不用重编**） |

**robot_lab 侧（只读、不部署改）：**

| 路径 | 作用 |
| --- | --- |
| `source/robot_lab/data/Robots/unitree/h1_description/mjcf/` | §8.1 用 `mujoco.viewer` 验加载；与 rl_sar_zoo 为 **两份拷贝**，改 sensor 时 **两边可同步** |
| `logs/.../exported/policy.pt` | §8.2 导出产物，再 cp 到 rl_sar `policy/h1/robot_lab/` |

---

### 8.1 步骤 1：MuJoCo 验证 MJCF 能加载

**做了什么：** 用厂商 `h1_description` 打开 MuJoCo viewer。

**怎么做：**

```bash
conda activate isaaclab
cd ~/project/robot_lab/source/robot_lab/data/Robots/unitree/h1_description
python -m mujoco.viewer --mjcf=mjcf/scene.xml
```

**验收：** 弹出 3D 窗口，可见 **Unitree H1 人形** mesh，无「找不到 mesh」报错。  
**姿态：** 可能 **趴地**——`scene.xml` 只加地面；`h1.xml` 为 `freejoint` + 默认关节角，viewer 跑物理后会倒。**不算失败**，本步只验 **模型能加载**。

**和训练对齐吗：** **不对齐也没关系**。这一步 **不涉及 policy**；Isaac 训用 **USD**，MuJoCo 用 **MJCF**（同厂商 `h1_description` 里另一套入口）。

**踩坑：**

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 在 `robot_lab` 根目录跑 viewer | 路径无 `mjcf/scene.xml` | 必须先 `cd` 到 `h1_description` |
| 终端「卡住」 | viewer 占满终端 | 关 **3D 窗口**；或 `pgrep -af mujoco.viewer` 后 `kill PID` |
| `grep "mujoco"` 有时搜不到进程 | 进程已退出；或应用 `pgrep -af mujoco.viewer` | 认准命令行含 `mujoco.viewer` 的那行 PID |

---

### 8.2 步骤 2：从 Flat-10 导出 `policy.pt`

**做了什么：** 用 **明确指定** 的 `model_1000.pt` 跑 `play.py`，生成 rl_sar 用的 JIT。

**为什么：** `exported/policy.pt` **不会自动** 对应 1000；`agent.yaml` 默认 `load_checkpoint: model_.*.pt` 会选 **字母序最大**（本 run 里曾到 `model_4000.pt`）。Flat-10 **sweet spot 在 ~1000**，4k 步态已过训。

**怎么做：**

```bash
cd ~/project/robot_lab
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Velocity-Flat-Unitree-My_H1 \
  --checkpoint logs/rsl_rl/unitree_my_h1_flat/2026-06-15_13-15-36/model_1000.pt \
  --num_envs 1 --headless
```

**验收：** 终端出现 `Loading model checkpoint from: .../model_1000.pt`；数秒后 `Ctrl+C` 即可。  
**产出路径：**

```text
logs/rsl_rl/unitree_my_h1_flat/2026-06-15_13-15-36/exported/policy.pt
```

**和什么对齐：** 与 **Flat-10 训练时 Actor 网络** 同权重；与 **第 7 章** `play` 验收为同一 ckpt。

---

### 8.3 步骤 3：编写 `rl_sar/policy/h1/robot_lab/config.yaml`

**状态：** [✓] 已按 Flat-10 改完（**待核对** `dof_vel_scale`、`action_scale` 个数）。

**文件作用：** 告诉 rl_sar **神经网络输入/输出如何映射到物理关节**——与 Isaac 的 `env.yaml` + `my_h1.py` 对齐；**不**描述 MJCF 网格或 ROS 话题名。FSM 进入 `RLFSMStateRLLocomotion` 时执行 `InitRL("h1/robot_lab")`，只读本文件 **`h1/robot_lab:`** 段。

**文件位置：**

```text
~/project/rl_sar/policy/h1/robot_lab/config.yaml
~/project/rl_sar/policy/h1/robot_lab/policy.pt   # 从 exported 拷入
```

**怎么建：** `cp policy/go2/robot_lab/config.yaml policy/h1/robot_lab/config.yaml`，再按下表改。

**为什么用 Go2 模板：** rl_sar 官方 locomotion 示例是 `go2/robot_lab`；字段结构（`observations`、`action_scale`、`rl_kp`…）可抄，**数值和维数不能抄**。

---

#### 8.3.1 对齐总表：三类参数

| 类别 | 必须和 Flat-10 / policy 一致 | 可先估、跑通再调 |
| --- | --- | --- |
| 观测维数、顺序、scale | `num_observations`、`observations`、`*_scale`（除见下） | — |
| 动作与关节零位 | `action_scale`、`default_dof_pos`、`num_of_dofs`、`joint_mapping` | — |
| 执行层 PD / 力矩 / clip | — | `rl_kp/kd`、`fixed_kp/kd`、`torque_limits`、`clip_actions` |

**对齐怎么查：**

| 训练侧（Isaac） | 部署侧（rl_sar） |
| --- | --- |
| `logs/.../params/env.yaml` → `observations.policy` | `observations:` 列表（名字见 `rl_sdk.cpp`） |
| `env.yaml` → `actions.joint_pos.scale` | `action_scale` |
| `my_h1.py` → `init_state.joint_pos` | `default_dof_pos` |
| `play` 日志 `Linear(in_features=…)` | `num_observations` |
| `h1.xml` → `<actuator>` 顺序 | `joint_mapping`、数组 19 维顺序 |

**Isaac 名 → rl_sar 名（`rl_sdk.cpp` 固定）：**

| env.yaml `policy` | config.yaml |
| --- | --- |
| `base_lin_vel` | `lin_vel` |
| `base_ang_vel` | `ang_vel` |
| `projected_gravity` | `gravity_vec` |
| `velocity_commands` | `commands` |
| `joint_pos` | `dof_pos` |
| `joint_vel` | `dof_vel` |
| `actions` | `actions` |

只读 **`policy:`** 段；**`critic:`** 重复出现是训练用，Sim2Sim **不管**。

---

#### 8.3.2 逐步修改记录（config.yaml）

**关节顺序（19 维数组都按此排列）：**

```text
左髋yaw, roll, pitch, 膝, 踝 | 右髋×5 | torso | 左肩×4 | 右肩×4
```

与 `rl_sar_zoo/h1_description/mjcf/h1.xml` 的 `<actuator>` 一致。

| 步 | 字段 | 改成什么 | 为什么 | 对齐来源 |
| --- | --- | --- | --- | --- |
| 1 | 顶层 key `h1/robot_lab:` | 勿用 `go2/robot_lab` | rl_sar 按 `robot_name/config_name` 读 yaml | rl_sar 约定 |
| 2 | `observations` | 首项加 `"lin_vel"` | Flat-10 policy **含** 线速度；Go2 模板无此项 | `env.yaml` `policy.base_lin_vel` |
| 3 | `num_observations` | `69` | 3+3+3+3+19×3；Flat 无 height_scan | 手算或 play 里 `in_features=69` |
| 4 | `num_of_dofs` | `19` | H1 可控关节数 | `h1_minimal.urdf` 数 `revolute` |
| 5 | `action_scale` | 19×`0.5` | 动作幅度 | `rough_env_cfg.py` `joint_pos.scale=0.5`；`env.yaml` `actions.joint_pos.scale` |
| 6 | `lin_vel_scale` 等 | `1.0`（`dof_pos` 已是 1.0） | Flat-10 去掉早期 2.0/0.25 等 | `env.yaml` 各 obs `scale: 1.0` |
| 7 | `default_dof_pos` | 19 个角（见下） | `dof_pos` 观测 = 当前角 − 默认角 | `my_h1.py` `init_state.joint_pos` |
| 8 | `clip_actions_*` | 各 19 个 ±100 | 维数对齐；±100≈不裁 | Go2 模板；训练 `clip_actions: null` |
| 9 | `rl_kp` | 腿 150/200、踝 20、腰 200、臂 40 | RL 模式 PD | `my_h1.py` `stiffness` |
| 10 | `rl_kd` | 腿 5、踝 4、腰 5、臂 10 | 与 kp 配对 | `my_h1.py` `damping` |
| 11 | `torque_limits` | 腿/腰/臂 300、踝 100 | 力矩上限 | `my_h1.py` `effort_limit_sim` |
| 12 | `fixed_kp/kd` | 暂同 `rl_kp/kd` | 非 RL 阶段站立 PD | 先跑通再调 |
| 13 | `joint_mapping` | `[0..18]` | MuJoCo actuator 下标 → policy 关节顺序 | 与 `h1.xml` `<actuator>` 一致 |

**当前 `config.yaml` 关键值（存档）：**

```yaml
h1/robot_lab:
  num_observations: 69
  observations: ["lin_vel","ang_vel", "gravity_vec", "commands", "dof_pos", "dof_vel", "actions"]
  num_of_dofs: 19
  action_scale: 19×0.5
  lin_vel_scale / ang_vel_scale / dof_pos_scale / commands_scale: 1.0
  dof_vel_scale: 0.05   # ← 待改为 1.0（Flat-10 训练为 1.0，Go2 残留）
  default_dof_pos: [0,0,-0.28,0.79,-0.52]×2, 0, [0.28,0,0,0.52]×2
  joint_mapping: [0,1,...,18]
  rl_kp/kd, fixed_kp/kd, torque_limits: 见 §8.3.2 步 9–11
```

**`default_dof_pos` 数值来源（弧度）：**

```text
髋y/r: 0, 0  |  髋pitch: -0.28  |  膝: 0.79  |  踝: -0.52  （左右腿同）
torso: 0
肩 pitch: 0.28, roll/yaw: 0, elbow: 0.52  （左右臂同）
```

**参数含义速查：**

| 参数 | 代表什么 |
| --- | --- |
| `joint_pos.scale` / `action_scale` | 目标角 = 默认角 + (policy输出 × scale) |
| `default_dof_pos` | 默认站姿；观测 `dof_pos` 为相对此的偏差 |
| `*_scale`（观测） | 拼进 69 维向量前的乘子 |
| `clip_actions_*` | policy 输出送电机前的上下限 |
| `rl_kp/kd` | RL 控制时关节 PD |
| `fixed_kp/kd` | 切 RL 前的固定 PD |
| `torque_limits` | 力矩饱和 |
| `joint_mapping` | MuJoCo 关节下标重排到训练顺序 |

---

#### 8.3.3 待核对项

| 项 | 状态 | 说明 |
| --- | --- | --- |
| `dof_vel_scale` | **应改为 `1.0`** | 当前文件仍为 `0.05`（Go2 残留）；Flat-10 `env.yaml` 为 1.0 |
| `action_scale` 个数 | **核对 19 个** | 当前文件有 18 个 `0.5`（少 1 个）→ 补全 |
| `policy.pt` 在位 | 运行时需要 | 可能因 `.gitignore` 不在 `git status` 里 |

---

### 8.4 步骤 4：编写 `policy/h1/base.yaml`

**状态：** [✓] 会话中按步改完；**须核对磁盘**：`ReadYaml` 读的是 yaml 里 **`h1:`** 段，顶层若仍为 `go2:` 会导致参数读空或启动异常。

**文件作用：** **仿真/实机公共底座**，与 `config.yaml` 分工如下：

| | `base.yaml` | `config.yaml` |
| --- | --- | --- |
| 谁读 | `rl_sim_mujoco` 启动：`ReadYaml("h1","base.yaml")` | FSM 切到 RL：`InitRL("h1/robot_lab")` |
| 内容 | `dt`、`decimation`、关节**名字**、`joint_mapping`、站起 PD `fixed_kp/kd` | obs 列表、scale、`action_scale`、RL 用 `rl_kp/kd`、`model_name` |
| 训练侧对应 | 仿真步长 ≈ Isaac `sim.dt`×`decimation` | `env.yaml` policy 段 + `my_h1.py` PD/零位 |

**怎么建：** `cp policy/go2/base.yaml policy/h1/base.yaml`，逐步修改：

| 步 | 字段 | 改法 |
| --- | --- | --- |
| 1 | 顶层 key | `go2:` → **`h1:`** |
| 2 | `num_of_dofs` | `12` → **`19`** |
| 3 | `joint_names` | 19 个 MJCF 关节名，顺序同 actuator：左髋 yaw/roll/pitch、膝、踝 → 右髋×5 → `torso_joint` → 左肩 pitch/roll/yaw、肘 → 右肩×4 |
| 4 | `joint_controller_names` | 与 `joint_names` 一一对应，规则 **`_joint` → `_controller`**（rl_sar/ROS 约定；MuJoCo 路径几乎不用，但要填） |
| 5 | `default_dof_pos` | **与 `config.yaml` 完全相同** 的 19 个数 |
| 6 | `joint_mapping` | `[0,1,2,...,18]`（policy 第 i 关节 = MuJoCo 第 i 个 actuator） |
| 7 | `fixed_kp` / `fixed_kd` / `torque_limits` | 与 `config.yaml` 相同 19 维 |
| 8 | `dt` / `decimation` | **保持** `0.005` / `4`（与 Isaac Flat-10 一致） |
| 9 | `wheel_indices` | **保持** `[]` |

**`joint_names` 完整列表（与 `h1.xml` actuator 一致）：**

```text
left_hip_yaw_joint, left_hip_roll_joint, left_hip_pitch_joint, left_knee_joint, left_ankle_joint,
right_hip_yaw_joint, right_hip_roll_joint, right_hip_pitch_joint, right_knee_joint, right_ankle_joint,
torso_joint,
left_shoulder_pitch_joint, left_shoulder_roll_joint, left_shoulder_yaw_joint, left_elbow_joint,
right_shoulder_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint, right_elbow_joint
```

---

### 8.5 步骤 5：`rl_sar_zoo/h1_description`

**状态：** [✓] 目录已就位；`scene.xml` include `h1.xml`。

**作用：** `rl_sim_mujoco h1 scene` 加载：

```text
rl_sar/src/rl_sar_zoo/h1_description/mjcf/scene.xml
  └─ include h1.xml（机器人 + actuator + sensor + mesh）
```

**怎么拷：**

```bash
cp -r ~/project/robot_lab/source/robot_lab/data/Robots/unitree/h1_description \
      ~/project/rl_sar/src/rl_sar_zoo/h1_description
```

厂商 `scene.xml` 只加地面/光照；**无** 19 关节 sensor 时 rl_sar 读 `sensordata` 会错（见 §8.6）。

---

### 8.6 步骤 5b：补 `h1.xml` 关节 sensor（rl_sar 必读）

**状态：** [✓] `rl_sar_zoo/.../h1.xml` 已按 Go2 布局补全 sensor。

**作用：** `rl_sim_mujoco.cpp` 的 `GetState` **不读** `qpos`，而是按固定下标读 `mj_data->sensordata`：

```text
sensordata[0 .. num_of_dofs-1]           关节位置
sensordata[num_of_dofs .. 2*num-1]       关节速度
sensordata[2*num .. 3*num-1]             关节力矩
sensordata[3*num .. 3*num+6]             IMU：四元数4 + 陀螺3
```

19 _dof 时共 **64** 维 sensordata；原厂商 `h1.xml` 仅 gyro+accel（6 维）→ 站起/PD **反馈错误**，FSM 显示 `Getting up completed` 但人形不动。

**怎么改：** 在 `h1.xml` 的 `<actuator>` 后，将原 `<sensor>` 整段替换为（**顺序必须与 `<motor>` 块一致**）：

1. 19× `<jointpos joint="..."/>`
2. 19× `<jointvel joint="..."/>`
3. 19× `<jointactuatorfrc joint="..."/>`
4. `<framequat objtype="site" objname="imu"/>`
5. `<gyro site="imu"/>`（勿再用旧版仅 accelerometer 的布局）

关节名列表同 §8.4 `joint_names`。改 MJCF **无需重编** C++，重启 `rl_sim_mujoco` 即可。

---

### 8.7 步骤 6：`fsm_h1.hpp` + 编译 + 首跑

**状态：** [✓] 已编译通过；`[FSMManager] Registered type: h1`；Passive 可进；行走 **待验收**。

#### 8.7.1 `fsm_h1.hpp`

**作用：** 键盘/手柄驱动的 **状态机**——无此文件则 `rl_sim_mujoco h1` 报 `Unsupported type: h1`。

| 状态 | 作用 | 按键（在 **终端** 输入，见下） |
| --- | --- | --- |
| `RLFSMStatePassive` | 关节 kd=8 阻尼 | 启动默认 |
| `RLFSMStateGetUp` | 2s 插值到 `default_dof_pos` | `0` / 手柄 A |
| `RLFSMStateRLLocomotion` | `InitRL("h1/robot_lab")` + `RLControl()` | 站起完成后 `1` / RB+上 |
| `RLFSMStateGetDown` | 插值趴下 | `9` / B |

**怎么建：**

```bash
cp src/rl_sar/fsm_robot/fsm_gr1t2.hpp src/rl_sar/fsm_robot/fsm_h1.hpp
```

全文替换：`GR1T2_FSM_HPP`→`H1_FSM_HPP`，`gr1t2_fsm`→`h1_fsm`，`Gr1t2FSMFactory`→`H1FSMFactory`，`"gr1t2"`→`"h1"`。  
在 `RLFSMStateRLLocomotion::Enter()`：**`rl.config_name = "robot_lab";`**（勿用 `legged_gym`）。

#### 8.7.2 `fsm_all.hpp`

增加：`#include "fsm_h1.hpp"`（触发文件末尾 `REGISTER_FSM_FACTORY`）。

#### 8.7.3 编译与运行

```bash
cd ~/project/rl_sar
./build.sh -mj
./cmake_build/bin/rl_sim_mujoco h1 scene
```

**首跑验收（已达成）：** 终端出现 `Registered type: h1`、`Entered passive mode`、`RL_Sim start`；MuJoCo 窗口 H1 落地。

**操作注意：**

- 键盘监听 **运行仿真的终端 STDIN**，不是 MuJoCo 3D 窗口；先点终端再按 `0`/`1`。
- spawn 后 **倒地正常**（MJCF 默认姿 + 重力）。
- `Joystick [/dev/input/js0] open failed` 可忽略（无手柄时用键盘）。
- `Getting up completed` 仅表示 **插值计时结束**；sensor 未补时视觉上可能仍不动。

**行走操作顺序：** 终端 `R` 重置 → `0` 站起 → `1` 进 RL → `WASD` / `QE` 给速度命令。

---

### 8.8 步骤 7（进行中）：行走验收与收尾

| 项 | 状态 |
| --- | --- |
| `dof_vel_scale: 1.0` | 待改 `config.yaml` |
| `action_scale` 19 个 | 待核对 |
| `base.yaml` 磁盘为 `h1:` | [✓] |
| sensor 补全后 `0` 站起可见关节运动 | [✓] 见 §8.9 |
| `1` 后 `InitRL` 无报错、能迈步 | 待用户复测（`0` 已可预载 policy） |
| 记录 play 视频/现象 | 待做 |

---

### 8.9 夜间调试：GetUp 秒倒 / 插值 ~1s（2026-06-23）

**现象（用户 Test B）：** `P` → `0` → 等待（不按 `1`）→ **<1s 倒地**；终端 `Getting up` 进度约 **1s** 而非 2s。

#### 根因（已确认）

| # | 根因 | 说明 |
| --- | --- | --- |
| 1 | **Keyframe ≠ 训练 `default_dof_pos`** | `h1.xml` `home` 曾用厂商腿姿 `(0,-0.4,0.8,-0.4)`、臂全 0；Isaac Flat-10 / `my_h1.py` 为 `(-0.28,0.79,-0.52)`、肩 `0.28/0.52`。Reset 后仍要做 2s 关节插值，人形在 **freejoint** 上不稳定。 |
| 2 | **`Interpolate()` 近零差分短路** | `rl_sdk.cpp`：若 `max\|start-target\|<0.1` rad，首帧把 `percent=1`（设计如此）。Keyframe 对齐后 **不再显示 2s 进度条**——属预期，不是计时 bug。 |
| 3 | **踝 `ctrlrange` 仅 ±40 Nm** | MJCF `<motor>` 踝限幅 40，训练 `effort_limit_sim=100`；PD 饱和，站立时踝无力矩余量。 |
| 4 | **开环 fixed PD 无法稳人形** | 四足 Go2 用 `pre_running_pos` 两阶段；H1 为 **freejoint 人形**，Test B（仅 fixed PD、无 policy）物理上不应期望长期站立。 |
| 5 | **终端 `0`+回车会暂停仿真** | `KeyboardInterface` 把 `\n` 映射为 `Enter` → `Simulation Stop`。打 `0` 后 **勿跟回车**；若已暂停再按一次 Enter 恢复。 |

**插值 ~1s 的解释：** 控制环 `dt=0.005`、2s 需 400 步，计时逻辑正确。旧 keyframe 与 default 腿差 ~0.12 rad（未触发 0.1 短路），但 **肩肘差 0.28 rad**；用户常在 **尚未满 2s 已倒地**，视觉上像「约 1s 就结束/倒了」。

#### 已做修改（rl_sar，已重编）

1. **`rl_sar_zoo/.../h1.xml`**：`home` keyframe 对齐 `my_h1.py` / `base.yaml` `default_dof_pos`；踝 `ctrlrange` **±100**。
2. **`fsm_h1.hpp`**：Passive→GetUp 时 Reset 后若 keyframe≈default 则 **`percent_getup=1` 跳过插值**；**预载 `InitRL(h1/robot_lab)`**，GetUp 完成态用 **`RLControl()` 零命令平衡**（无需先按 `1`）；`RLLocomotion::Enter` 若已 init 则跳过重复加载。
3. **`policy/h1/base.yaml`**：`fixed_kp/kd` 与训练 `rl_kp/kd` 对齐（fallback）。

**自动化冒烟：** `printf '0'`（无回车）管道启动 → 日志含 `GetUp: RL policy loaded` 与 `Successfully loaded Torch model`；无 `Getting up` 进度（跳过插值）。GUI 长稳测试需本机显示器。

#### 用户晨间复测步骤

```bash
cd ~/project/rl_sar && cmake --build cmake_build -j$(nproc)
./cmake_build/bin/rl_sim_mujoco h1 scene
```

1. 焦点在 **启动仿真的终端**（非 MuJoCo 窗口）。
2. **`0` 单键、不要回车**（避免暂停仿真）。
3. 应看到 `GetUp: RL policy loaded for zero-command balance.`；机器人用 **policy 零命令** 平衡，无需按 `1`。
4. 稳定后按 `W` 试走；或按 `1` 进入 `RLLocomotion`（已 init 则秒切）。
5. 若仍秒倒：记录终端全文 + 是否误触 Enter；再查 obs 对齐（`lin_vel` body 系、`dof_vel_scale` 等）。

---

### 遇到的问题

| 现象 | 原因 / 处理 |
| --- | --- |
| 以为 `exported/policy.pt` 必是 1000 | 默认 checkpoint 正则选 **最新** `model_*.pt` → 须 **显式** `--checkpoint .../model_1000.pt` 重导出 |
| `ps grep mujoco` 结果不一致 | 窄 grep 仅进程瞬间；`viewer` 匹配过宽会扫到 Cursor sandbox；用 **`pgrep -af mujoco.viewer`** |
| MuJoCo viewer 终端卡死 | 关 3D 窗；勿只靠 `Ctrl+C` |
| 编辑器多光标 / 中键列编辑无效 | Linux 上 Alt 常被窗口管理器占用；yaml 数组建议 **整段替换** |
| rl_sar 无 H1 | 官方 zoo 仅有 g1/go2/…，H1 需 **自建** policy + FSM + description |
| `Unsupported type: h1` | 未建 `fsm_h1.hpp` 或未 `#include` 进 `fsm_all.hpp` 或未重编 |
| 按 `0` 终端有 FSM 切换但 **人形不动** | 原 `h1.xml` **无 jointpos/vel sensor**；rl_sar 把陀螺仪当关节角 → PD 失效；按 §8.6 补 sensor |
| `Getting up completed` 仍趴着 | 同上；或键盘未打在 **终端**（读 STDIN 非 MuJoCo 窗口） |
| spawn 后倒地 | **正常**；MJCF 默认姿态 + 重力，不等于加载失败 |
| `joint_controller_names` 是什么 | **非训练产物**；rl_sar ROS 命名约定，`xxx_joint`→`xxx_controller`；MuJoCo 路径主要靠 `joint_mapping` |
| `base.yaml` 与 `config.yaml` 区别 | base=仿真步长/关节名/mapping/站起 PD；config=obs/action/RL PD/加载 `policy.pt` |
| `ReadYaml` 读不到参数 | `base.yaml` 顶层必须是 **`h1:`**（与 `robot_name` 一致）；`config.yaml` 顶层必须是 **`h1/robot_lab:`** |
| `0` 后秒倒 / 插值 ~1s | 见 **§8.9**：keyframe 与训练姿不一致、踝力矩限幅、开环 PD 不能稳人形；`0`+**回车** 会 **暂停仿真** |
| Test B（不按 `1`）期望站立 | 已改为 GetUp 后 **自动预载 policy + 零命令 RLControl**；纯 fixed PD 仍不能长期稳人形 |

### 知识点

- **Sim2Sim 对齐的是「向量怎么拼、动作怎么缩放」**，不是 USD 与 MJCF 文件格式一致。
- **MJCF** 在厂商包 `h1_description/mjcf/`；**Isaac 训练不用 MJCF**；rl_sar 跑仿真时 **从 `rl_sar_zoo` 加载**。
- **policy 维数：** 只认 `policy:` 观测项；关节维数 = URDF/MJCF **revolute 个数（19）**。
- **names 映射** 以 `rl_sar/.../rl_sdk.cpp` 的 `if (observation == "...")` 为准，不能填 `base_lin_vel` 这种 Isaac 名。
- **`POLICY_DIR`** 在编译时写入（`rl_sar/policy`）；改 yaml/pt **不用重编**；改 FSM **要重编**；改 MJCF **不用重编**。
- **三类配置链：** Isaac 训 → `policy.pt` + `env.yaml`；部署 → `config.yaml`（网络 I/O）+ `base.yaml`（仿真底座）+ `h1.xml`（物理与 sensor 布局）。

---

## 第 9 章 实机部署（rl_sar → 真机）

### 实践步骤

尚未进行。

### 遇到的问题

（尚未进行。）

### 知识点

- 交付常翻车：joint 顺序、action scale、观测与仿真不一致。

---

## 附录 A：G1 → H1 差异速查

| 项目 | G1 | H1 |
| --- | --- | --- |
| DOF | 29 | 19 |
| 脚 link | `*_ankle_roll_link` | `*_ankle_link` |
| 腰 | `waist_yaw_joint` 等 | `torso_joint` |
| 踝 | pitch + roll | 单 `ankle_joint` |
| 站立高度 z | ~0.76 | ~1.05 |
| action scale | 可按关节算 | 可先统一 0.25 |

---

## 附录 C：官方 H1 Flat baseline（参考入口）

官方 H1 Flat baseline（**robot_lab 自训**）只作为参考对象。  
具体训练信息、曲线现象和结论见 **第 10 章**「官方 H1 Flat baseline（robot_lab 自训）」。

---

## 附录 D：官方 H1 Rough baseline（参考入口）

官方 H1 Rough baseline（**robot_lab 自训**）只作为参考对象。  
具体训练信息、play 现象和结论见 **第 10 章**「官方 H1 Rough baseline（robot_lab 自训）」。

---

## 附录 E：Isaac Lab 官方 H1 预训练 play（参考入口）

**Isaac Lab Nucleus 预训练**（非 robot_lab 仓库自带权重）。  
加载方式、与 rsl_rl 5.3 格式兼容、play 现象及与 Flat-3 对比见 **第 10 章**「Isaac Lab 官方 H1 预训练 play（Nucleus，对照上限）」。

---

## 第 10 章 实验记录（集中版）

> 每个实验单独成节。每节只记录：**改动（相对上一轮）/ 现象 / 验收结论**。不放训练/play 命令。  
> 图片已从会话 assets 归档到 `docs/images/`（含 Flat-3、Isaac Lab 预训练 play 等），按实验/环境分类放回对应小节。每轮 **训完贴 TensorBoard、play 贴视口 + debug 终端**。

### Run 1 / Run 01：初版 My_H1 训练

**改动（初始实验）：**

| 类别 | 内容 |
| --- | --- |
| Asset | 初版 `my_h1.py`，由 G1/官方 asset 改来，URDF 指向 H1 |
| 尚未稳定项 | `joint_pos`、torso PD、actuator 分组、自碰撞等尚未完全调稳 |
| Env | 初始 Rough 训练配置 |

**现象：**

- 约 300 iter，未训满。
- 总 reward / episode length 有一定表现，但没有单独记录脚离地分项。

![TensorBoard：mean episode length / mean reward（Run 1）](./images/tensorboard_episode_reward.png)

![TensorBoard：Episode_Termination/time_out（Run 1）](./images/tensorboard_termination_timeout.png)

![TensorBoard：Episode_Termination/terrain_out_of_bounds（Run 1）](./images/tensorboard_termination_out_of_bounds.png)

**验收结论：**

- play 基本躺地，**FAIL**。
- 旧 asset 条件不足，不能作为可走基线。

### Run 2 / Run 02：新 asset 长训

**改动（相对 Run 1）：**

| 类别 | Run 1 | Run 2 |
| --- | --- | --- |
| `joint_pos` | 初版 / 未稳 | 改为官方 H1 站立角 |
| `torso_joint` | PD/effort 不完整 | 补 `effort/stiffness/damping` |
| 自碰撞 | 曾开启或不稳定 | `enabled_self_collisions=False` |
| Env | 初始配置 | 基本沿 Run 1 |
| 训练规模 | ~300 iter | 长训约 20k step |

**现象：**

- mean reward / episode length 收敛，time_out 很高。
- `feet_air_time≈0`，`track_lin_vel` 没形成走路趋势。

![play：多 env 基本倒地（Run 1/2）](./images/play_my_h1_rough_multi_env.png)

![play：单 env 趴在尖刺/石块格（Run 1/2）](./images/play_my_h1_rough_prone.png)

![play：镜头跟到别的 env 格子（Run 2）](./images/run02_play_wrong_camera_env.png)

![TensorBoard：mean reward / episode length 收敛（Run 2）](./images/run02_tensorboard_convergence_episode_reward.png)

![TensorBoard：mean_reward 平台期（Run 2）](./images/run02_tensorboard_mean_reward.png)

![TensorBoard：time_out≈0.99（Run 2）](./images/run02_tensorboard_upward_contact.png)

![TensorBoard：Episode_Reward/upward（Run 2）](./images/run02_tensorboard_feet_air_time.png)

![TensorBoard：feet_air_time≈0（Run 2）](./images/run02_tensorboard_timeout.png)

**验收结论：**

- 躺赢 / 撑局，**FAIL**。
- 长训不能解决不会走。

### Run 3 / Run 03：小命令 + base height + feet_air 加权

**改动（相对 Run 2）：**

| 类别 | Run 2 | Run 3 |
| --- | --- | --- |
| policy `base_lin_vel` | 关闭：`None` | 恢复观测 |
| 命令范围 | 大范围 | 缩小到小命令，`lin_vel_x/y`、`ang_vel_z` 约 `(-0.3, 0.2)` |
| `base_height_l2.weight` | `0` | `-1.0`，`target_height=0.9` |
| `feet_air_time.weight` | `0.25` | `0.50` |
| `undesired_contacts` | `0` | 不变 |

**现象：**

- 约 521 iter 主动停。
- `feet_air_time` 仍在 ~0.001 平台。
- `track` 有改善，但不能代表迈步。

![TensorBoard：~124 step，feet_air 刚开局（Run 3）](./images/run03_tensorboard_feet_air_early_124.png)

![TensorBoard：track_lin_vel 分项可>1（Run 3）](./images/run03_tensorboard_track_lin_vel_gt1.png)

![TensorBoard：~521 step 主动停，feet_air≈0.001（Run 3）](./images/run03_tensorboard_stop_521.png)

**验收结论：**

- 未 play；按曲线判断 **FAIL/无效**。
- 没有突破脚离地问题。

### Run 4 / Run 04：加强非脚触地惩罚

**改动（相对 Run 3）：**

| 类别 | Run 3 | Run 4 |
| --- | --- | --- |
| `undesired_contacts.weight` | `0` | `-0.5` |
| 触地对象 | 未惩罚 | 惩罚非脚 link 触地 |
| 其它 | 小命令、base height、`feet_air_time=0.5` | 不变 |

**现象：**

- 约 800 iter。
- `feet_air_time` 略好但仍 ~0.001。
- undesired_contacts 惩罚生效，episode length 未明显崩。

![TensorBoard：feet_air_time 略好（Run 4）](./images/run04_tensorboard_feet_air_time.png)

![TensorBoard：track_lin_vel 改善（Run 4）](./images/run04_tensorboard_track_lin_vel.png)

**验收结论：**

- 未 play；**FAIL/无效**。
- 只是略有改善，未突破。

### Run 5 / Run 05：asset/PD 稳定后重训

**改动（相对 Run 4）：**

| 类别 | Run 4 | Run 5 |
| --- | --- | --- |
| `hip_pitch` 初始角 | 仍在调整 | 约 `-0.27` |
| 踝 `armature` | 曾有错误大值 | `0.01` |
| 踝 effort | 较低 | `100` |
| 腿/腰 PD | 仍不完整/不稳 | 髋膝约 `200`，腰/腿 PD 补齐 |
| 自碰撞 | 可能干扰 | `enabled_self_collisions=False` |
| `undesired_contacts.weight` | `-0.5` | `-0.2` |
| `feet_air_time` | weight `0.5`，threshold `0.4` | 不变 |
| 命令 | 小范围全向 | 不变 |

**现象：**

- `feet_air_time` 升到约 ~0.0036。
- episode length 接近满。
- 非脚触地减少。

![play：spawn 后直立（Run 5）](./images/run05_play_fall_seq_03_falling.png)

![play：开始后仰 / 动态失衡倒向（Run 5）](./images/run05_play_fall_seq_04_falling.png)

<!-- 归档文件名 run05_play_fall_seq_01/02/05 内容与 play 不符（代码编辑 / nvidia-smi / 四足），已移除引用 -->

![TensorBoard：feet_air 新 run 开局（Run 5）](./images/run05_tensorboard_feet_air_start.png)

![TensorBoard：~1200 step feet_air≈0.003（Run 5）](./images/run05_tensorboard_feet_air_1200.png)

![TensorBoard：~1200 step track（Run 5）](./images/run05_tensorboard_track_1200.png)

![TensorBoard：~1200 step undesired_contacts（Run 5）](./images/run05_tensorboard_undesired_1200.png)

![TensorBoard：~1374 step feet_air≈0.0031（Run 5）](./images/run05_tensorboard_feet_air_1374.png)

![TensorBoard：~1558 step feet_air≈0.0036（Run 5）](./images/run05_tensorboard_feet_air_1558.png)

![TensorBoard：Run 5 改动后曲线组 1](./images/run05_tensorboard_curves_set_01.png)

![TensorBoard：Run 5 改动后曲线组 2](./images/run05_tensorboard_curves_set_02.png)

![play：以右脚为圆心慢慢转身（Run 5/6）](./images/run05_play_spin_on_right_foot.png)

**验收结论：**

- 能站，右脚类似支点，左脚微抬点地并转圈。
- **FAIL**，不是双足交替走。

### Run 6 / Run 06：关转向、只前进、提高 feet_air

**改动（相对 Run 5）：**

| 类别 | Run 5 | Run 6 |
| --- | --- | --- |
| `lin_vel_x` | 小范围全向 | `(0, 0.3)` 只往前 |
| `lin_vel_y` | 可有侧向 | `(0, 0)` |
| `ang_vel_z` | 可转向 | `(0, 0)` |
| `feet_air_time.weight` | `0.5` | `1.0` |
| `feet_air_time.threshold` | `0.4` | 不变 |
| `undesired_contacts.weight` | `-0.2` | 不变 |

**现象：**

- 约 2805 step。
- `feet_air_time` 仍是 10^-3 量级。
- track 有改善，undesired_contacts 更好。

![TensorBoard：Run 6 曲线 1](./images/run06_tensorboard_curves_01.png)

![TensorBoard：Run 6 曲线 2](./images/run06_tensorboard_curves_02.png)

**验收结论：**

- 仍转圈 / 叉腿，不形成前向交替步态。
- **FAIL**。

### Run 7 / Run 07：切到平地 plane

**改动（相对 Run 6）：**

| 类别 | Run 6 | Run 7 |
| --- | --- | --- |
| 地形 | Rough | `terrain_type="plane"` |
| `terrain_generator` | 开 | `None` |
| `terrain_levels` | 开 | `None` |
| 命令 / reward | 只前进、`feet_air=1.0`、`undesired=-0.2` | 不变 |

**现象：**

- 约 15k step。
- track 很高，undesired_contacts 接近 0。
- `feet_air_time` 后期崩到很低。

![play：双脚点地往前蹭/小跳，不抬腿（Run 7）](./images/run07_play_zombie_shuffle.png)

![play debug：cmd=0.3 但几乎不挪（Run 7）](./images/run07_play_debug_no_move.png)

**验收结论：**

- 双脚贴地蹭 / 小跳，有位移但不抬脚。
- **FAIL**。长训把抬脚趋势训没了。

### Run 8 / Run 08：降低 track 权重

**改动（相对 Run 7）：**

| 类别 | Run 7 | Run 8 |
| --- | --- | --- |
| `track_lin_vel_xy_exp.weight` | `3.0` | `1.5` |
| 地形 | plane | 不变 |
| 命令 | 只前进 | 不变 |
| `feet_air_time` | weight `1.0`，threshold `0.4` | 不变 |
| `undesired_contacts.weight` | `-0.2` | 不变 |

**现象：**

- 约 3507 step。
- track 不再刷满。
- `feet_air_time` 峰值约 ~0.002 后回落。

![TensorBoard：Run 8 track=1.5 曲线 1](./images/run08_tensorboard_curves_01.png)

![TensorBoard：Run 8 track=1.5 曲线 2](./images/run08_tensorboard_curves_02.png)

![TensorBoard：Run 8 feet_air + track（改 track=1.5）](./images/run08_tensorboard_feet_air_track_01.png)

![TensorBoard：Run 8 曲线（改 track=1.5）](./images/run08_tensorboard_feet_air_track_02.png)

![play debug：站桩几乎不前进（Run 8）](./images/run08_play_debug_standing.png)

**验收结论：**

- 站住 / 微挪，几乎不走。
- **FAIL**。

### Run 9 / Run 09：提高 feet_air threshold

**改动（相对 Run 8）：**

| 类别 | Run 8 | Run 9 |
| --- | --- | --- |
| `feet_air_time.threshold` | `0.4` | `0.6` |
| `track_lin_vel_xy_exp.weight` | `1.5` | 不变 |
| 地形 / 命令 | plane / 只前进 | 不变 |
| `feet_air_time.weight` | `1.0` | 不变 |

**现象：**

- 约 3000 step。
- 曲线和 Run 8 接近。
- `feet_air_time` 没有突破。

![TensorBoard：Run 9 vs Run 8 对比曲线](./images/run09_tensorboard_vs_run08.png)

![TensorBoard：Run 9 threshold=0.6 曲线](./images/run09_tensorboard_curves.png)

**验收结论：**

- 站住不动。
- **FAIL**。单改 threshold 无效。

### Run 10：关闭 base_height 奖励

**改动（相对 Run 9）：**

| 类别 | Run 9 | Run 10 |
| --- | --- | --- |
| `base_height_l2.weight` | `-1.0` | `0` |
| `base_height_l2.target_height` | `0.9` | 保留但 weight 为 0 |
| 其它 | threshold `0.6`、track `1.5`、plane、只前进 | 不变 |

**现象：**

- 约 9281 step。
- `feet_air_time` 早期峰值约 ~0.002 后持续下跌。
- 长训无益。

![TensorBoard：Run 10 关 base_height 曲线](./images/run10_tensorboard_curves.png)

![play：站桩不动（Run 10）](./images/run10_play_standing.png)

![TensorBoard：~3069 step feet_air 下跌（Run 10）](./images/run10_tensorboard_feet_air_decline.png)

![TensorBoard：~2192 step 是否该停（Run 10）](./images/run10_tensorboard_stop_decision_2192.png)

**验收结论：**

- 站住不动。
- **FAIL**。关闭 base height 无效。

### Run 11：关闭 policy base_lin_vel 观测

**改动（相对 Run 10）：**

| 类别 | Run 10 | Run 11 |
| --- | --- | --- |
| policy `base_lin_vel` | 开 | `None`，关闭 |
| policy 输入维度 | 69 | 66 |
| `base_height` | `0` | 不变 |
| `track` / threshold / plane / 命令 | `1.5` / `0.6` / plane / 只前进 | 不变 |

**现象：**

- 约 3820 step。
- track 改善，但 `feet_air_time` 仍是 10^-3 量级。

![play：站桩不动（Run 11）](./images/run11_play_standing.png)

![play：轻微晃动、脚不抬（Run 11）](./images/run11_play_slight_shake.png)

**验收结论：**

- `model_2800` 站住 + 微晃。
- **FAIL**。关 `base_lin_vel` 无效。

### Run 12 R1：统一 action scale

**改动（相对 Run 11）：**

| 类别 | Run 11 | Run 12 R1 |
| --- | --- | --- |
| action scale | `UNITREE_My_H1_ACTION_SCALE` | 统一 `0.25` |
| 其它 | `base_lin_vel=None`、threshold `0.6`、track `1.5`、plane | 不变 |

**现象：**

- `feet_air` 峰后回落。
- 最佳 ckpt 约 `model_3150`。

**验收结论：**

- `vx≈0`、`Δx≈5cm`，站桩。
- **FAIL**。

### Run 12 R2：加硬踝 PD

**改动（相对 Run 12 R1）：**

| 类别 | Run 12 R1 | Run 12 R2 |
| --- | --- | --- |
| ankle stiffness | `20` | `50` |
| ankle damping | `4` | `10` |
| action scale | `0.25` | 不变 |
| 其它 | Run 12 R1 | 不变 |

**现象：**

- `feet_air` 回落。
- 最佳 ckpt 约 `model_1800`。

**验收结论：**

- `vx≈0`、`Δx≈-3cm`，站桩/后退。
- **FAIL**。

### Run 12 R3：加硬 hip/knee

**改动（相对 Run 12 R2）：**

| 类别 | Run 12 R2 | Run 12 R3 |
| --- | --- | --- |
| `hip_pitch` stiffness | `200` | `350` |
| `knee` stiffness | `200` | `350` |
| ankle PD | `50/10` | 不变 |
| action scale | `0.25` | 不变 |

**现象：**

- 约 1366 step 后 play。
- 最佳 ckpt 约 `model_1300`。

**验收结论：**

- `vx≈0`、`Δx≈-2cm`，站桩。
- **FAIL**。

### Run 12 R3 误续训

**改动（相对 Run 12 R3）：**

| 类别 | Run 12 R3 | 误续训 |
| --- | --- | --- |
| cfg | 无新改动 | 无新改动 |
| 流程 | 应 play 后结束本轮 | 错误从 `model_1350` resume |
| 训练上限 | 应进入下一轮 | 又传 `max_iterations=10000` |

**现象：**

- 跑到 `11349/11350`。
- 不是按“训→play→改一项”的流程。

![TensorBoard：Run 12 R3 误续训 ~1650 step](./images/run12_r3_resume_tensorboard_1650.png)

**验收结论：**

- **无效实验**。
- 不能作为调参结论。

### Run 13 R1：降低 feet_air threshold

**改动（相对 Run 12 末态）：**

| 类别 | Run 12 末态 | Run 13 R1 |
| --- | --- | --- |
| `feet_air_time.threshold` | `0.6` | `0.4` |
| `base_lin_vel` | `None` | 不变 |
| action scale | `0.25` | 不变 |
| track | `1.5` | 不变 |
| ankle / hip/knee | `50/10` / `350` | 不变 |

**现象：**

- 训满 3500。
- `feet_air` 峰约 ~0.00192。

![play：Run 13 R1 下蹲后摔倒](./images/run13_r1_play_fall.png)

**验收结论：**

- 100 step 前摔倒，后续倒地滑行。
- **FAIL**。`threshold=0.4` 分支变坏。

### Run 13 R2：降低 track 权重

**改动（相对 Run 13 R1）：**

| 类别 | Run 13 R1 | Run 13 R2 |
| --- | --- | --- |
| `track_lin_vel_xy_exp.weight` | `1.5` | `1.0` |
| `feet_air_time.threshold` | `0.4` | 不变 |
| 其它 | Run 13 R1 | 不变 |

**现象：**

- 训满 3500。
- `feet_air` 峰约 ~0.00192。

![play：Run 13 R2 开局摔倒](./images/run13_r2_play_fall.png)

**验收结论：**

- 100 step 前摔倒，后续倒地滑行。
- **FAIL**。不沿此分支继续。

### Run 13 R3 test：坏分支上回退 hip/knee

**改动（相对 Run 13 R2）：**

| 类别 | Run 13 R2 | Run 13 R3 test |
| --- | --- | --- |
| `hip_pitch` stiffness | `350` | `200` |
| `knee` stiffness | `350` | `200` |
| `feet_air_time.threshold` | `0.4` | 不变 |
| `track` | `1.0` | 不变 |

**现象：**

- 只训练到 `105/3500`，不完整。

**验收结论：**

- **无效实验**。
- 与 Run 14 R1 不等价，因为它继承了 `threshold=0.4`、`track=1.0` 的坏分支。

### Run 14 R1：干净分支单测 hip/knee 回退

**改动（相对 Run 13 坏分支 / 回到 Run 12 末态）：**

| 类别 | Run 13 坏分支 | Run 14 R1 |
| --- | --- | --- |
| `feet_air_time.threshold` | `0.4` | 回到 `0.6` |
| `track_lin_vel_xy_exp.weight` | `1.0` | 回到 `1.5` |
| `hip_pitch` stiffness | `350` | `200` |
| `knee` stiffness | `350` | `200` |
| action scale | `0.25` | 不变 |
| ankle PD | `50/10` | 不变 |

**现象：**

- 训满 3500。
- `feet_air` 峰约 ~0.00198。

**验收结论：**

- 不摔，但 `Δx≈0.008`，双脚接触，稳定站桩。
- **FAIL**。

### Run 14 R2：回退 ankle PD

**改动（相对 Run 14 R1）：**

| 类别 | Run 14 R1 | Run 14 R2 |
| --- | --- | --- |
| ankle stiffness | `50` | `20` |
| ankle damping | `10` | `4` |
| hip/knee | `200` | 不变 |
| action scale | `0.25` | 不变 |

**现象：**

- 训满 3500。
- `feet_air` 峰约 ~0.00165。

**验收结论：**

- 不摔，`Δx≈-0.041`，双脚接触。
- **FAIL**，站桩。

### Run 14 R3：回退 action scale

**改动（相对 Run 14 R2）：**

| 类别 | Run 14 R2 | Run 14 R3 |
| --- | --- | --- |
| action scale | `0.25` | `UNITREE_My_H1_ACTION_SCALE` |
| ankle PD | `20/4` | 不变 |
| hip/knee | `200` | 不变 |
| threshold / track | `0.6` / `1.5` | 不变 |

**现象：**

- 训满 3500。
- `feet_air` 峰约 ~0.00205。

**验收结论：**

- 摔倒，`base_z≈0.058`。
- **FAIL**。动作尺度回退导致不稳。

### Run 14 R4：降低 joint_pos_penalty

**改动（相对 Run 14 分支）：**

| 类别 | 改动前 | Run 14 R4 |
| --- | --- | --- |
| `joint_pos_penalty.weight` | `-1.0` | `-0.5` |
| 目的 | 默认姿态约束较强 | 给抬腿 / 摆腿更多空间 |

**现象：**

- 训满 3500。
- `feet_air` 峰约 ~0.00247。

**验收结论：**

- 不摔，`Δx≈0.030`，双脚接触。
- **FAIL**，站桩。

### Run 14 R5：提高 feet_air 权重

**改动（相对 Run 14 R4）：**

| 类别 | Run 14 R4 | Run 14 R5 |
| --- | --- | --- |
| `feet_air_time.weight` | `1.0` | `1.5` |
| `feet_air_time.threshold` | `0.6` | 不变 |
| `joint_pos_penalty.weight` | `-0.5` | 不变 |

**现象：**

- 训满 3500。
- `feet_air` 峰约 **~0.00384**。
- `low_ceiling=False`，是 Run 14 里最好的曲线。

![play：轻微抬腿趋势后后仰（Run 14 R5）](./images/run14_r5_play_attempt_lift_fall.png)

**验收结论：**

- 肉眼看有轻微晃动和抬腿趋势，但像抬不起来，随后后仰。
- 自动 debug play 曾被 CUDA/NVML 阻塞，**未完全验收**。
- 当前最值得复验。

### Run 14 R6：关闭 heading command

**改动（相对 Run 14 分支）：**

| 类别 | 改动前 | Run 14 R6 |
| --- | --- | --- |
| `heading_command` | 默认 | `False` |
| `rel_heading_envs` | 默认 | `0.0` |
| 目的 | 减少 heading 随机 / 朝向命令干扰 | 只看前进控制是否改善 |

**现象：**

- 训满 3500。
- `feet_air` 峰约 ~0.00282。

**验收结论：**

- 不摔，`Δx≈0.139`，双脚接触。
- **FAIL**，仍站桩/贴地。

### Run 15 R1：提高 feet_air 权重（好分支）

**改动（相对 Run 14 R5 基线 / 去掉 R6 heading）：**

| 类别 | Run 14 R6 末态 | Run 15 R1 |
| --- | --- | --- |
| `feet_air_time.weight` | `1.0`（R5 的 1.5 已丢） | **`2.0`** |
| `feet_air_time.threshold` | `0.6` | 不变 |
| `track_lin_vel_xy_exp.weight` | `1.5` | 不变 |
| `joint_pos_penalty.weight` | `-0.5` | 不变 |
| `heading_command` | `False`（R6） | **恢复默认** |
| 地形 / 命令 | plane / 只前进 | 不变 |
| 腿 / 踝 PD | hip/knee `200/8`，ankle `20/4` | 不变 |

**现象：**

- **2048 env**，目录 `2026-06-10_02-08-02`，训满 3500。
- `feet_air` 早期峰约 **~0.006**（step ~750），后期塌到 **~0.0004**。
- `track` 平台 **~1.3**；典型「高 track + 低 feet_air」。
- **1024 env**，目录 `2026-06-10_03-00-50`，训至 **~15000** iter。
- `feet_air` step ~4000 峰约 **~0.005**，最终 **~0.0001**；`track` 平台 **~1.31**。
- **1024 vs 2048**：结局同类（贴地赢），2048 按 iter 收敛更快；**不改变能不能抬脚**。

**验收结论：**

- **2048 play**（`model_750`）：轻微抬腿趋势后后仰，仍 **FAIL**。
- **1024 play**（`model_4000`）：开局上身晃，脚不抬，后站桩；比 `model_11350` 略好。
- **1024 play**（`model_11350`，末期 feet_air 尖峰 ckpt）：几乎无前进欲望，双脚全程贴地，**更差**。
- **FAIL**。加大 `feet_air` 只抬高早期峰值，后期仍塌；末期尖峰 ckpt 不可信。

### Run 15 R2：降低 track 权重

**改动（相对 Run 15 R1）：**

| 类别 | Run 15 R1 | Run 15 R2 |
| --- | --- | --- |
| `track_lin_vel_xy_exp.weight` | `1.5` | **`1.0`** |
| `feet_air_time.weight` | `2.0` | 不变 |
| `feet_air_time.threshold` | `0.6` | 不变 |
| `joint_pos_penalty.weight` | `-0.5` | 不变 |
| 其它 | plane、只前进、`base_lin_vel=None`、action `0.25` | 不变 |

**说明：** Run 13 R2 也做过 `track=1.0`，但在 **threshold=0.4 坏分支** 上；本轮是在 Run 15 好分支上单测降 track。

**现象：**

- 目录 `2026-06-10_10-04-46`，训满 **3500**（1024 env）。
- step 2500 前 `feet_air` 低；**末期上冲**，raw 约 **~0.0076**，smoothed 约 **~0.0043**（`declined=False`，和 R1 末期崩塌不同）。
- `track` 约 **~0.71** 仍在涨（未刷到 1.3）；`undesired_contacts` 约 **~-0.11**。
- **误 resume** 目录 `2026-06-10_12-54-57`：`feet_air` 从 **~0.007 掉回 ~0.002**，`track` 涨到 **~0.82**——长训把脆弱抬脚趋势训没，**不应 resume**。

**验收结论：**

- play `model_3450`（首轮 3500 末）：**上身晃得更厉害**，但 `foot_z≈0.062` 不变、`left/right_contact` 全程 True，**脚不抬**。
- 比 Run 7「僵尸蹭地」更差：Run 7 至少有贴地前移；R2 是 **站桩扭动型 FAIL**。
- 曲线比 R1 健康，但 **未转化为 play 迈步**；resume 退步。
- **FAIL**（曲线部分有效 / play 无效）。

### Run 15 R3：髋膝 damping 对齐官方（8→5）

**改动（相对 Run 15 R2，`my_h1.py`）：**

| 类别 | Run 15 R2 | Run 15 R3 |
| --- | --- | --- |
| `hip_pitch` damping | `8` | **`5`**（官方同值） |
| `knee` damping | `8` | **`5`** |
| hip_roll / hip_yaw / torso damping | `5` | 不变 |
| `rough_env_cfg` | track `1.0`，feet_air `2.0`，threshold `0.6` | **不变** |

**现象：**

- 目录 `2026-06-10_13-28-01`，训满 **3500**。
- step ~1800 前 `feet_air` 低；之后 **持续上冲**，峰 smoothed **~0.00346**（step ~2520），末 **~0.0028**。
- **`declined=False`**，末期未像 R1/R2 塌到 0.000x；是 Run 15 里 **末期最稳** 的 `feet_air` 曲线。

**验收结论：**

- play `model_2500.pt`：**往后小幅僵尸跳**（cmd 向前但 `vx` 为负），约 **三步后后仰摔倒**。
- step 1 debug：`vx≈-0.31`，`left/right_contact=False`（离地但是后跳，非迈步）。
- 曲线末期 `feet_air` 虽稳，但 play 是 **不稳定后跳 → 摔**，不是双足走。
- **FAIL**（比 R2 站桩晃更差：有非法动态 + 摔倒）。

### Run 15 R4：髋膝 damping 折中（5→6）

**改动（相对 Run 15 R3，`my_h1.py`）：**

| 类别 | Run 15 R3 | Run 15 R4 |
| --- | --- | --- |
| `hip_pitch` damping | `5` | **`6`** |
| `knee` damping | `5` | **`6`** |
| hip_roll / hip_yaw / torso damping | `5` | 不变 |
| `rough_env_cfg` | track `1.0`，feet_air `2.0`，threshold `0.6` | **不变** |

**现象：**

- 目录 `2026-06-10_14-20-31`，1024 env，训满 **3500**。
- `feet_air` 峰 smoothed **~0.00364**（step ~2870），末 **~0.0028**；`declined=True`（末期有回落）。
- 曲线介于 R3（末期稳）与 R1/R2（末期塌）之间；**未转化为稳定步态**。

**验收结论：**

- play `model_2850.pt`（用户亲验）：**进去微微下蹲，然后往后仰倒摔倒**。
- debug：step 1 `base_z≈1.05` 正常；step 100～200 `base_z≈0.12～0.18`，`pitch≈-0.83`，`fallen=True`（后仰倒，非前栽）。
- 与 R3 同属 **后仰 FAIL**；R3 后跳感更明显，R4 先微蹲再倒，摔法略不同但均未迈步。
- **FAIL**。当前 cfg 停在本轮（damping 6）；原计划的 damping 7 未开训（用户手动叫停）。

### Run 15 R5：官方奖励配方 + My_H1 asset/PD

**目的：** 在 **Run 15 R4 PD 不变** 的前提下，将 `rough_env_cfg` 奖励 / 命令 / 地形对齐 **官方 H1 Rough**，隔离「自定义奖励 + plane + 只前进」与 asset/PD 的影响。

**改动（相对 Run 15 R4）：**

| 类别 | Run 15 R4 | Run 15 R5 |
| --- | --- | --- |
| `track_lin_vel_xy_exp.weight` | `1.0` | **`3.0`**（官方） |
| `feet_air_time.weight` | `2.0` | **`1.0`**（官方） |
| `joint_pos_penalty.weight` | `-0.5` | **`-1.0`**（官方） |
| `undesired_contacts.weight` | `-0.2` | **`0`**（官方） |
| `joint_mirror.weight` | `-0.1` | **`0`**（官方） |
| `joint_deviation_torso_l1` | `-0.2` / `torso_joint` | **`-0.1`** / `torso_joint`（权重官方；关节名 My_H1 适配） |
| `joint_deviation_hip/arms` 正则 | `.*hip_yaw.*` 等 | **权重 -0.2 同官方**；正则保留 `*_joint` 后缀适配（官方 `.*_hip_yaw` 在 My_H1 URDF 不匹配） |
| `joint_torques_l2` / `joint_acc_l2` joint_names 过滤 | 仅腿关节 | **移除**（官方无过滤） |
| `base_height_l2.target_height` | `0.9` | **`0`**（官方） |
| 命令 `lin_vel_x/y`, `ang_vel_z` | `(0,0.3)` / `(0,0)` / `(0,0)` | **`(-1,1)` 全向**（官方） |
| 地形 | `plane`（`terrain_generator=None`） | **rough**（恢复父类 `ROUGH_TERRAINS_CFG` + `terrain_levels` curriculum） |
| `my_h1.py` PD | hip_pitch/knee damping `6`，其余同 R4 | **不变** |

**说明：** 全新目录 **`2026-06-10_21-14-59`**；**勿 `--resume`**。失败尝试 `2026-06-10_20-49-43`、`2026-06-10_21-07-39`（均仅 `model_0.pt`）：前者 hip 正则 `.*_hip_yaw` 与 My_H1 URDF 不匹配报错，后者用户 ~20 s 内叫停；已改回 `.*hip_yaw.*` 等适配后 fresh 重训。

**现象：**

- 目录 **`2026-06-10_21-14-59`**，1024 env，训满 **5000** iter（`Curriculum/terrain_levels` 已激活，确认 rough）。
- 早期 `illegal_contact` 高（~93%）、`ep_length` 低；中后期 `feet_air` **持续上涨**，末期未像 R1/R2 塌回 0.000x。
- step ~4831：`feet_air` smoothed **~0.0029**（raw ~0.0024），仍在缓升；`track_lin_vel_xy_exp` smoothed **~0.59**（~1000 step 后平台 ~0.55–0.65）。
- 对照：官方 H1 Rough `feet_air≈0.016`、`track≈1.44`；Run 14 R5 plane 峰 **~0.00384**。本轮曲线是 Run 15 里 **末期最稳的 feet_air 上涨**，但绝对值仍远低于官方。
- 典型组合：**中等 track + 缓慢上涨的 feet_air**，非 R7「高 track + 低 feet_air」贴地赢形态。

![TensorBoard：Run 15 R5 feet_air + track（~4831 step）](./images/run15_r5_tensorboard_feet_air_track.png)

**验收结论：**

- play **`model_4800.pt`**（`feet_air` Smoothed 峰值附近）：
  - step 1：`fallen=False`，`base_z≈1.14`，`vx≈0.4`，双足 `contact=False`（reset 后短暂离地）。
  - step 100～500：**`fallen=True` 全程**（`play.py` 调试判据：`base_z<0.55` 或 `|roll/pitch|>0.8`）；`cmd` 全向变化；`|action|≈1.3～1.9`。
- **<100 step 内扑倒**；之后 **play 不会因摔倒而 reset**（与训练局终止不同），仿真步数仍走到 500，人一直趴地，策略继续输出动作——**不是站起来继续走**，只是倒地后的贴地蹭动/乱扭。
- 摔法非 R4「微蹲后仰」；更像 Run 13 的 **早摔 + 倒地贴地**。
- **FAIL**。官方奖励 + rough + 全向命令 **未转化为 play 迈步**；**排除**「仅自定义 plane 奖励配方」是主因，问题更可能在 **My_H1 asset/PD/观测**。
- 实验价值：曲线健康 ≠ 会走；勿用末期 ckpt 盲信（峰值 ckpt 仍摔）。

![play：Run 15 R5 step 1 站姿正常（model_4800）](./images/run15_r5_play_step1_standing.png)

![play debug：Run 15 R5 step 100～500 fallen=True（model_4800）](./images/run15_r5_play_debug_fallen.png)

### Run 15 R6：官方 PD / effort 对齐 H1_CFG

**目的：** 在 **Run 15 R5 官方奖励 + rough 不变** 前提下，将 `my_h1.py` 的 PD / effort / 初始角 / solver 对齐 **`isaaclab_assets` `H1_CFG`**，隔离「自定义 PD / armature / 腿 effort」是否为 play 早摔主因。仍用 **URDF**（非官方 USD）。

**改动（相对 Run 15 R5，`my_h1.py`）：**

| 类别 | Run 15 R5 | Run 15 R6 |
| --- | --- | --- |
| 髋膝 damping | `6` | **`5`**（官方） |
| 腿 effort | 220 / torso 200 | **300**（官方） |
| 腿 / 踝 armature | 有（G1 公式 + 踝 0.01） | **去掉** |
| 腿 velocity_limit | 有 | **去掉** |
| 臂 PD | 5020 电机公式 | **40 / 10**（官方） |
| 臂 effort | 25 | **300** |
| solver_position_iterations | 8 | **4** |
| `rough_env_cfg` | R5 官方奖励 | **不变** |

**现象：**

- 目录 **`2026-06-11_01-36-43`**，1024 env，训满 **~5000** iter。
- step ~4999：`feet_air` smoothed **~0.0036**（raw ~0.0031），比 R5 **~0.0029** 略高；曲线 **尖刺多**（500 step 后上下跳），不如 R5 平滑。
- `track` smoothed **~0.53**，略低于 R5 **~0.59**；仍远低于官方 Rough **~1.44**。

![TensorBoard：Run 15 R6 feet_air + track（~4999 step）](./images/run15_r6_tensorboard_feet_air_track.png)

**验收结论：**

- play **`model_4800.pt`**（`feet_air` Smoothed 峰值附近），`--hold_seconds` 暂停后开策略：
  - **全向命令（与训时一致）：** step 1 `cmd≈[-0.40,-0.84,0.54]`，`|action|≈0.99`，`fallen=False`；用户观感：**spawn 后像被一股力推**（实为 **`debug_vis` 速度命令箭头**，非 `push_robot` 外力）、**左脚像钉在 rough 格子上**、**右脚踝/髋 uncontrollable 转动**后失衡倒。
  - **play 临时改 `cmd=(0.3,0,0)` 只前进：** step 1 `vx≈0.48`、`yaw≈-0.61`（spawn 朝向仍随机），**摔法/时机与全向命令无明显差别**（用户：**感觉没区别**）。
  - step 100 起 **`fallen=True`**（与 R5 同类）；play 不 reset，趴地至 step 500。
- **FAIL**。相对 R5 **曲线略好、play 无肉眼改善**；**排除**「自定义 PD / armature / 腿 effort」是 play 早摔主因。
- 剩余差异：**URDF vs 官方 USD**、策略本身（后文 R7 平地 play 已 **排除**「仅 rough 卡脚」为主因）。

![play：Run 15 R6 step 1 全向命令 + 速度箭头（model_4800）](./images/run15_r6_play_step1_full_cmd.png)

![play：Run 15 R6 失衡拧转（model_4800）](./images/run15_r6_play_tumble.png)

![play：Run 15 R6 step 1 仅前进 cmd=0.3（model_4800，与全向摔法无差别）](./images/run15_r6_play_step1_forward_cmd.png)

### Run 15 R7：PPO `entropy_coef` 对齐官方

**改动（相对 Run 15 R6）：**

| 类别 | Run 15 R6 | Run 15 R7 |
| --- | --- | --- |
| `entropy_coef` | `0.008` | **`0.01`**（官方 H1 Rough） |
| 奖励 / PD / 地形 / 命令 | R6 | **不变** |

**现象：**

- 目录 **`2026-06-11_09-06-32`**，1024 env，训满 **~25000** iter。
- step ~12145：`feet_air` smoothed **~0.003**，`track` **~0.68**；`feet_air` 尖刺多。
- step ~24999：`feet_air` smoothed **~0.0029**（raw ~0.0028）；`track` smoothed **~0.72** — Run 15 里 **track 最高**，但 `feet_air` 仍 ~0.003 量级，远低于官方 Rough **~0.016**。

![TensorBoard：Run 15 R7 feet_air + track（~24999 step）](./images/run15_r7_tensorboard_feet_air_track.png)

**验收结论（rough play）：**

- play **`model_12000.pt`**（`feet_air` 峰值附近），`--hold_seconds` + `--follow`：
  - step 1：`fallen=False`，`base_z≈1.05`，双足 `contact=False`。
  - step 100：`fallen=True`，`base_z≈0.26`，**`right_contact=True`、`left_contact=False`**。
  - 用户观感：**右腿一撇、直接坐/瘫在地上**（不对称坐地，非 R4 后仰、非 R5 纯扑倒）。
- **FAIL**。`entropy_coef` 对齐后 **曲线 track 变好，play 仍秒级/百步级摔**。

![play：Run 15 R7 step 1（model_12000，rough）](./images/run15_r7_play_step1.png)

![play：Run 15 R7 右腿撇坐地（model_12000，rough）](./images/run15_r7_play_sit_right_leg.png)

**平地 play 对照（诊断，非新训）：**

- 同一 ckpt **`model_12000.pt`**，`play.py` 加 **`--plane`**（平地覆盖，task 仍为 My_H1 Rough，obs 维不变）。
- 结果：**与 rough 相同** — 右腿撇、坐地；step 1 `base_z≈1.05`、`fallen=False`，随后仍摔。
- **排除**「仅 rough 地形 / 一脚卡石块」是 play 早摔主因；问题在 **策略 + URDF 本体**（与平地/rough 无关）。

![play：Run 15 R7 平地 `--plane` 仍右腿坐地（model_12000）](./images/run15_r7_play_plane_sit.png)

### Run 15 R5～R7：官方 cfg 对齐隔离小结

| 轮次 | 相对上轮改动 | 曲线 | play | 隔离结论 |
| --- | --- | --- | --- | --- |
| **R5** | 官方奖励 + rough + 全向 | `feet_air` 缓涨 ~0.003 | 百步内扑倒 | 排除「仅自定义 plane 奖励」 |
| **R6** | 官方 PD / effort（`my_h1.py`） | `feet_air` ~0.0036，尖刺多 | 与 R5 **无肉眼差别** | 排除「自定义 PD / armature」 |
| **R7** | `entropy_coef=0.01` | `track` ~0.72（最高） | 右腿撇坐地；**平地一样** | 排除「PPO 探索」与「仅 rough 卡脚」 |

**共同 FAIL 形态：** 曲线 `track` / `feet_air` 可涨，**play 均不能站走**；勿再堆 reward/PD/PPO iter。

**剩余主嫌：** **URDF vs 官方 USD**、训坏策略。

**spawn / zero 对照（R7 后）：**

| 对象 | zero（Flat / Rough） | 训后 play |
| --- | --- | --- |
| **My_H1 URDF** | 秒仰倒 / 膝顶 / 多向倒 | <100 step 坐地 |
| **官方 H1 USD** | **Flat 也秒倒**（与 My_H1 差不多） | baseline ckpt：**叉腿、靠右脚蹭地挪**（step 200 `fallen=False`） |

→ zero 差不多 **不能** 证明只有 URDF 坏；**训后官方明显好于 My_H1** → Run 16 换 USD 仍有意义。

### Run 16：My_H1 task + 官方 `H1_MINIMAL_CFG`（USD）

**目的：** 在 **R7 cfg 不变**（官方奖励 / PD 在 USD 内置 / PPO）下，**只换 robot asset**：`UNITREE_My_H1_CFG`（URDF）→ **`H1_MINIMAL_CFG`**（USD），验证 play 能否接近官方 baseline。

**改动（`unitree_my_h1/rough_env_cfg.py`）：**

| 类别 | R7（URDF） | Run 16 |
| --- | --- | --- |
| `scene.robot` | `UNITREE_My_H1_CFG` | **`H1_MINIMAL_CFG`** |
| `foot_link_name` | `.*_ankle_link` | **`.*ankle_link`**（USD） |
| 奖励关节正则 | `*_joint` 后缀 | **官方 USD 关节名**（`torso` 等） |
| task id / 奖励权重 / PPO | R7 | **不变** |
| 开关 | — | `_USE_OFFICIAL_H1_USD = True`（改回 `False` 恢复 URDF） |

**现象：**

- 目录 **`2026-06-12_01-48-27`**，1024 env，训满 **~3500** iter。
- step ~3444：`feet_air` smoothed **~0.015**（≈ 官方 Rough **~0.016**）；`track` smoothed **~1.05**（URDF R7 仅 ~0.72）。
- 训练曲线 **首次达到官方 Rough 量级**。

![TensorBoard：Run 16 feet_air + track（~3444 step）](./images/run16_tensorboard_feet_air_track.png)

**验收结论：**

- play **`model_3400.pt`**（`--plane --follow --hold_seconds 8`）：
  - step 1：`fallen=False`，`base_z≈1.05`。
  - step 100：`fallen=False`，`vx≈0.2`，`left/right_contact=True`，有位移。
  - 用户观感：**与官方 baseline 一样** — 双脚岔开、**依靠右脚蹭地挪动**（叉腿瘸步，非标准双足走）。
- **隔离实验 PASS**：换 USD 后训+play **接近官方**；**证实 My_H1 URDF 是 R5～R7 训/play 失败的主因**。
- **步态验收仍 FAIL**（与官方 baseline 同级）：不算标准双足走；近期可 **继续用 USD 线**（`_USE_OFFICIAL_H1_USD=True`），URDF 需另开工程修 mesh/惯性/碰撞。

![play：Run 16 叉腿右脚蹭地（与官方同类，model_3400）](./images/run16_play_right_foot_shuffle.png)

### Run 17：`joint_mirror=-0.1`（USD 线）

**目的：** 在 Run 16 基础上 **只加** 左右关节对称惩罚，看能否改善交替步态。

**改动（相对 Run 16）：**

| 类别 | Run 16 | Run 17 |
| --- | --- | --- |
| `joint_mirror.weight` | `0` | **`-0.1`** |
| `mirror_joints` | 占位 | **`[["left_(hip\|knee\|ankle).*", "right_(hip\|knee\|ankle).*"]]`** |
| 其余 | — | **不变** |

**现象：**

- 目录 **`2026-06-12_02-39-50`**，~3500 iter。
- step ~3499：`feet_air` smoothed **~0.020**（略高于 Run 16 **~0.015**）；`track` smoothed **~1.01**。

**验收结论：**

- play：仍 **叉腿 / 单脚主导蹭地**，观感与 Run 16 同级。
- **步态 FAIL**；`joint_mirror` 管的是 **左右关节角接近**（且实现只比第一对关节），**不是** 左右脚交替；不宜再加权重。
- `joint_mirror` 已改回 **`0`**（与官方 H1 一致）。

### Run 18：步态强化包（USD + 前后命令）

**目的：** 在 Run 16 上打包改奖励与命令，推 **往前走 + 抬脚 + 左右脚时间均衡**（非单变量实验）。

**改动（相对 Run 16）：**

| 类别 | Run 16 | Run 18 |
| --- | --- | --- |
| `feet_air_time.weight` | `1.0` | **`2.0`** |
| `feet_air_time_variance` | 未开 | **`weight=-0.5`** |
| `joint_deviation_hip_l1` | `-0.2` | **`-0.3`**（压 hip_yaw/roll 岔腿） |
| `joint_mirror` | `0` | **`0`** |
| `feet_height` / `feet_height_body` | `0` | **`0`**（未启用；不管抬多高） |
| 命令 `lin_vel_x` | `(-1,1)` 全向 | **`(-1,1)`**（可前进可后退） |
| 命令 `lin_vel_y` / `ang_vel_z` | `(-1,1)` | **`(0,0)`**（无横移、无转向） |
| 训练长度 | ~3500 iter | **~25000 iter** |

**现象：**

- 目录 **`2026-06-12_03-36-58`**。
- step ~24999：`feet_air` smoothed **~0.052**（≈ 官方 Flat **0.05+** 量级）；`track` smoothed **~1.89**；`feet_air_time_variance` smoothed **~-0.006**（惩罚项生效）。

**验收结论：**

- play **`model_24000.pt`**（须 **`--num_envs 1`**；`--plane --follow --hold_seconds 8`）：
  - `fallen=False`，**明显往前位移**（`vx` 可达 **~1.1**，优于 Run 16 蹭地挪）。
  - step 100 附近曾有 **单脚支撑**（一脚 `z≈0.10`、另一脚 `z≈0.07` 着地）。
  - 脚离地仅 **~3–5 cm**（微抬）；**右脚抬幅仍大于左脚**，观感仍瘸。
- **训练 PASS**；**步态部分 PASS**（能前走、微抬脚）/ **对称仍 FAIL**（非拟人步态）。
- 认知：`feet_air_time` 只奖 **离地多久**，不奖 **抬多高**；`feet_height_body` 等人形 cfg 默认关。

### Run 19：Rough 步态强化 + `feet_height`（打包，相对 Run 18）

**目的：** 在 Run 18 思路上加强左右均衡、只前进、试抬脚高度；仍在 **Rough task** 训练。

**改动（相对 Run 18）：**

| 类别 | Run 18 | Run 19 |
| --- | --- | --- |
| `feet_air_time_variance` | `-0.5` | **`-1.0`** |
| `lin_vel_x` | `(-1,1)` | **`(0, 0.5)`** |
| `feet_height` | `0` | **`-1.0`**，`target_height=0.08` |
| `feet_height_body` | `0` | `0` |
| 训练长度 | ~25000 iter | **~12700 iter**（`2026-06-12_10-03-07`） |

**现象（TensorBoard，~12700 step）：**

- `feet_air` smoothed **~0.026**（低于 Run 18 **~0.052**）。
- `track` smoothed **~2.18**；`variance` smoothed **~-0.019**。

**验收结论（play）：**

- **`--plane` + `--forward`**：能走，左右幅度更接近，但 **抬脚更低、更贴地**（相对 Run 18）。
- **Rough 无 `--plane` + `--forward`**（`model_12700.pt`）：`cmd=[0.3,0,0]` 下 **挪 ~0.8 m 后 `vx≈0` 卡死**；双脚扒地（脚 `z≈0.25` 地形抬高）。
- **Rough 验收 FAIL**；策略偏 **平地贴地滑步**，换起伏地形即失效。
- 注释 `play.py` 强制前进 **无效** 根因：`heading_command=False` 未取消 + `rel_standing`；已改为 CLI **`--forward`**（每步钉死命令）。

### Run 20：`h1_minimal.urdf` + Flat-10 奖励（`2026-06-16_02-06-53`）— **URDF 线 FAIL**

**目的：** 在 **Flat-10 MDP / 奖励数字不变** 前提下，验证 **Isaac 风格简化碰撞 URDF**（`h1_minimal.urdf`，23→3 collision）能否复现 Flat-10 @1000 步态；**不先改奖励**。

**改动（相对 Flat-10）：**

| 类别 | Flat-10 | **Run 20** |
| --- | --- | --- |
| `scene.robot` | **`H1_MINIMAL_CFG`（USD）** | **`UNITREE_My_H1_CFG`（URDF）** |
| URDF 文件 | — | **`h1_minimal.urdf`**（脚+躯干碰撞） |
| `_USE_OFFICIAL_H1_USD` | `True` | **`False`** |
| 奖励 / events / commands / PPO | Flat-10 | **相同**（落盘 `params/env.yaml` 已核对） |
| `foot_link_name` | `.*ankle_link` | `.*_ankle_link`（命名适配，权重同） |

**训练：**

| 项 | 内容 |
| --- | --- |
| Run | **`2026-06-16_02-06-53`** |
| 命令 | `train.py --task RobotLab-Isaac-Velocity-Flat-Unitree-My_H1 --headless --max_iterations=35000` |
| 须 **fresh** | **勿 resume** Flat-10 ckpt |

**现象（TB @~1038）：**

- `feet_air` smoothed **~0.14**（Flat-10 @1000 **~0.05** → **偏高**，像 Flat-9 早期）。
- `track_lin_vel_xy_exp` smoothed **~0.98**（跟速分项已高）。

![TensorBoard：Run 20 @~1038 feet_air + track（用户截图）](./images/run20_tensorboard_1038.png)

**验收结论（play `model_1000.pt`，@0.5，`--forward --follow`）：— FAIL**

- **跟速 PASS**：step 100 `vx≈0.43`，step 200 `vx≈0.50`。
- **对称 FAIL**：`left_contact=True` **持续**，`right_contact=False`；右脚常驻空中。
- **抬脚 FAIL**：左 `z≈0.063` 贴地，右 `z≈0.15～0.31`（单脚蹦跳，非交替）。
- **与 Flat-4～9「一脚钉地」同类**；**不是 iter 不够**（勿续训到 35k）。

![play：Run 20 `model_1000` @0.5 — 单脚抬 FAIL + debug（用户截图）](./images/run20_play_1000_single_leg.png)

**小结：**

- **奖励不是主因**（同 Flat-10 数字，USD 能走、URDF 不能）。
- **简化碰撞数量不够**：`h1_minimal.urdf` ≠ `h1_minimal.usd` 物理行为；与 Run 16 结论一致。
- **决策：路线 B** — 先修 **asset（URDF 对齐 USD）**，奖励保持 Flat-10；**勿先堆 `feet_air` / `variance` 补丁**。

**路线 B 下一步（Run 20 后续）：**

1. ~~**spawn 对照**~~ **已做（`compare_h1_spawn.py`，Flat env，zero action，200 step）：**

| step | 资产 | `base_z` | `pitch` | `left_z` / `right_z` | 接触 | `fallen` |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | USD | 1.048 | 0.000 | 0.140 / 0.140 | 无 | False |
| 1 | URDF | 1.048 | 0.000 | 0.140 / 0.140 | 无 | False |
| 50 | USD | 0.888 | -0.050 | 0.073 / 0.073 | 双 | False |
| 50 | URDF | **0.856** | +0.033 | **0.062** / **0.062** | 双 | False |
| 100 | USD | **0.998** | **0.000** | 0.091 / 0.091（离地） | 无 | False |
| 100 | URDF | **0.833** | **-0.317** | 0.062 / 0.062 | 双 | False |
| 200 | USD | 0.910 | -0.105 | 0.070 / 0.070 | 双 | False |
| 200 | URDF | **0.864** | **-0.210** | 0.062 / 0.062 | 双 | False |

**spawn 解读：**

- **初值一致**（`base_z=1.048`、脚 `z=0.14`）→ spawn 高度 / 初始角不是主差异。
- **落地后 URDF 更塌、更前倾**：`base_z` 低 ~5–17 cm；step 100 `pitch≈-0.32`（USD 近 0）。
- **脚着地高度**：URDF 稳定在 **z≈0.062**，USD **0.070–0.091** → 碰撞 box / 导入可能与 USD 脚高不一致。
- **URDF 导入警告（P0 前）**：`imu_link` / `d435_*` / `mid360_link` / `logo_link` 被 **merge 进 `torso_link`**（带惯性）→ 躯干 COM/惯量与 `h1_minimal.usd` 不同。
- **zero 不能单独解释 play 单脚**（两者 200 step 均未 `fallen`），但 URDF **动力学更软、更前倾**，训后更易掉进「单脚蹭地」局部最优。

**URDF 优先修项（路线 B-2）：**

| 优先级 | 项 | 动作 |
| --- | --- | --- |
| P0 | 传感器 link 合并 | ~~从 `h1_minimal.urdf` 删掉 `imu`/`d435`/`mid360`/`logo`~~ **已做（2026-06-16）**；导入 **无 merge 警告**；zero spawn 前倾仍存 → 继续 P1 |
| P1 | 脚碰撞 box | ~~对照 Isaac `h1_minimal.usd` 对齐脚 collision~~ **已做（2026-06-16）**：`origin z: -0.05 → -0.061`（spawn 脚 z 0.062→**0.073**，与 USD 一致）；`pitch` 仍差 → 可 fresh 训验 |
| P2 | 躯干/全身惯性 | ~~核对 `torso_link`~~ **已做（2026-06-16）**：`torso` 本就一致；**全身 20 link** 质量/COM/惯量对齐 USD（总质量 59.3→**51.4 kg**）；spawn `pitch@100` **-0.35→-0.27** |
| P3 | 再训 | 对齐后 **fresh** Flat-10 @1000 play；**勿 resume Run 20** |

### Run 20 路线 B：P0 删传感器 link（`2026-06-16`）

**目的：** 消除 URDF 导入时 fixed 传感器子 link **merge 进 `torso_link`**，使躯干惯量接近 `h1_minimal.usd`。

**改动（`h1_minimal.urdf`）：**

| 删除项 | 类型 |
| --- | --- |
| `imu_link` + `imu_joint` | fixed → torso |
| `logo_link` + `logo_joint` | fixed → torso（含 visual mesh） |
| `d435_left_imager_link` + joint | fixed → torso |
| `d435_rgb_module_link` + joint | fixed → torso |
| `mid360_link` + `mid360_joint` | fixed → torso |

腿 / 臂 / 躯干碰撞与关节 **未动**；`my_h1.py` / 奖励 / `_USE_OFFICIAL_H1_USD=False` **未动**。

**验证（`compare_h1_spawn.py`，P0 后复跑）：**

| 项 | P0 前 | **P0 后** |
| --- | --- | --- |
| 导入 merge 警告 | 有（imu/d435/mid360/logo） | **无** |
| zero @100 `pitch`（URDF） | -0.317 | **-0.317**（未改善） |
| zero @100 `base_z`（URDF） | 0.833 | **0.833** |
| 脚着地 `z`（URDF） | 0.062 | **0.062** |

**结论：**

- **P0 PASS（导入层面）**：merge 警告消除，躯干不再被传感器子树污染。
- **spawn 动力学仍与 USD 有差**：前倾、脚低未变 → P1 脚碰撞仍值得做，但 **可先 P0 后 fresh 训看策略是否改善**。
- **Run 20 ckpt 作废**：`02-06-53` 在 P0 前 asset 上训的，**勿 resume**。

**下一步（两条线可并行）：**

1. **P0 后 fresh 训** → 见 **Run 21**（`02-45-30`）。
2. **P1** — 若 Run 21 @1000 仍 FAIL，再对照 `h1_minimal.usd` 对齐脚 collision box。

---

### Run 21：P0 后 fresh 训（`2026-06-16_02-45-30`）— **FAIL**

**目的：** P0 改完 `h1_minimal.urdf` 后 **立刻 fresh 训**，验证删传感器 link 是否足以改善步态（奖励仍 Flat-10，**不等 P1**）。

**相对 Run 20：**

| 类别 | Run 20 | **Run 21** |
| --- | --- | --- |
| URDF | P0 前（含传感器 link） | **P0 后**（已删 imu/d435/mid360/logo） |
| Run 目录 | `02-06-53` | **`02-45-30`** |
| 奖励 / MDP | Flat-10 | **相同** |
| 关系 | — | **fresh**；**勿 resume** `02-06-53` |

**现象（TB @~4563）：**

- `feet_air` smoothed **~0.27**（Run 20 @1k **~0.14**；Flat-10 @1k **~0.05** → **更高、更糟**）。
- `track_lin_vel_xy_exp` smoothed **~0.92**（跟速分项高）。

![TensorBoard：Run 21 @~4563 feet_air + track（用户截图）](./images/run21_tensorboard_4563.png)

**验收结论（play `model_2000.pt`，@0.5，`--forward --follow`）：— FAIL**

- **跟速 PASS**：step 100 `vx≈0.56`，step 200 `vx≈0.41`。
- **对称 FAIL**：`left_contact=True` **持续**，`right_contact=False`；与 Run 20 **同形态**（左脚撑地、右脚常驻空中）。
- **P0 单独不够**：删传感器 link 未改变策略局部最优。

![play：Run 21 `model_2000` @0.5 — 单脚抬 FAIL + debug（用户截图）](./images/run21_play_2000_single_leg.png)

**小结：**

- **P0 必要但不充分**（merge 警告没了，步态没变）。
- **勿续训到 35k**（`feet_air` 已 0.27，比 Run 20 更饱和）。
- **下一步：P1** — 对照 `h1_minimal.usd` 对齐脚 collision box，再 **fresh**（Run 21 ckpt **勿 resume**）。

---

### Run 20 路线 B：P1 脚碰撞对齐 — **spawn 验收 PASS（2026-06-16）**

**目的：** 对齐 URDF 与 USD 的 **脚着地高度**（Run 20 spawn：URDF 脚 `z≈0.062` vs USD `0.073`）。

**改动（`h1_minimal.urdf`，`left/right_ankle_link`）：**

| 项 | P1 前 | **P1 后** |
| --- | --- | --- |
| collision `origin` | `xyz="0.05 0.0 -0.05"` | **`xyz="0.05 0.0 -0.061"`** |
| collision `box size` | `0.28 0.03 0.024` | **不变** |
| 推导依据 | — | spawn @step50 脚 z 差 **11mm**（0.073−0.062） |

**验收方法：** `scripts/tools/compare_h1_spawn.py`，Flat env，zero action，200 step，`--asset both --headless`。

```bash
python scripts/tools/compare_h1_spawn.py \
  --task RobotLab-Isaac-Velocity-Flat-Unitree-My_H1 \
  --asset both --num_envs 1 --steps 200 --headless
```

**验收数据（P1 后复跑，2026-06-16）：**

| step | 资产 | `base_z` | `pitch` | 脚 z (L/R) | 接触 | `fallen` |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | USD / URDF | 1.048 | 0.000 | 0.140 / 0.140 | 无 | False |
| 50 | USD | 0.888 | -0.051 | **0.073** / 0.073 | 双 | False |
| 50 | URDF（P1前） | 0.856 | +0.033 | **0.062** / 0.062 | 双 | False |
| 50 | **URDF（P1后）** | **0.865** | +0.049 | **0.073** / 0.073 | 双 | False |
| 100 | USD | 0.998 | **0.000** | 0.091 / 0.091 | 无 | False |
| 100 | URDF（P1前） | 0.833 | -0.317 | 0.062 / 0.062 | 双 | False |
| 100 | **URDF（P1后）** | 0.835 | -0.355 | **0.073** / 0.073 | 双 | False |
| 200 | USD | 0.910 | -0.105 | 0.070 / 0.070 | 双 | False |
| 200 | URDF（P1前） | 0.864 | -0.211 | 0.062 / 0.062 | 双 | False |
| 200 | **URDF（P1后）** | **0.903** | **-0.132** | **0.073** / 0.073 | 双 | False |

**验收结论：**

| 分项 | 结果 | 说明 |
| --- | --- | --- |
| **脚着地高度** | **PASS** | step50 脚 z **0.073**，与 USD **一致**（P1前 0.062） |
| **`base_z`** | 部分改善 | step50 0.856→**0.865**（USD 0.888）；仍略低 |
| **`pitch` 前倾** | **未 PASS** | step100 仍 **≈-0.35**；step200 **-0.13** 优于 P1前 **-0.21**，仍差于 USD **-0.11** |
| **步态 / play** | **未验** | spawn 只验物理落地；**须 Run 22 训+play** 才能验是否还单脚抬 |

**小结：** P1 **spawn 验收 PASS**（脚高对齐）；**不能**据此认定走路已修好。Run 20/21 ckpt **作废**；下一步 **Run 22 fresh 训** @1000 play。

---

### Run 22：P0+P1 后 fresh 训（`2026-06-16_04-08-18`）— **FAIL**

**目的：** P0（删传感器 link）+ P1（脚碰撞 `origin z: -0.05→-0.061`）后 **fresh 训**，验证 play 是否脱离单脚局部最优。

**相对 Run 21：**

| 类别 | Run 21 | **Run 22** |
| --- | --- | --- |
| URDF | P0 后 | **P0+P1 后** |
| Run | `02-45-30` | **`04-08-18`** |
| 奖励 / MDP | Flat-10 | **相同** |

**现象（TB @~23556）：**

- `feet_air` smoothed **~0.31**（Flat-10 @1k **~0.05**；Run 21 @4.5k **~0.27** → **更饱和**）。
- `track_lin_vel_xy_exp` smoothed **~0.93**。

![TensorBoard：Run 22 @~23556 feet_air + track（用户截图）](./images/run22_tensorboard_23556.png)

**验收结论（play 对比）：**

| ckpt | 观感 | debug 要点 |
| --- | --- | --- |
| **`model_500.pt`** | 左腿像 **支撑脚**，抬不起来；**高频碎步** | step100：左 z=0.125 **着地**，右 z=0.073；step200：左 z=0.160 **仍着地**，右 z=0.072 |
| **`model_2000.pt`** | 频率变慢；**右脚抬高** | step100：左 z=0.073 **着地**，右 z=**0.189** 悬空 |

![play：Run 22 `model_500` — 左脚钉地 + 高频（用户截图）](./images/run22_play_500_early.png)

![play：Run 22 `model_2000` — 左脚钉地 + 右脚大摆（用户截图）](./images/run22_play_2000_single_leg.png)

**演化解读：**

- **@500**：已是 **左脚支撑偏好** + 右腿贴地；高频 = 早期碎抖蹭 `feet_air`，**不是**交替步态。
- **@2000**：收敛为 Run 20/21 同类捷径 — **一脚钉地 + 一脚大摆**；频率降 = 局部最优固化。
- **@23k TB**：`feet_air≈0.31` → **勿续训**；P0+P1 **不足以**让 URDF 线走出 Flat-10 形态。

**小结：**

- **路线 B P0+P1 训后仍 FAIL**；Run 22 ckpt **勿 resume**。
- **下一步 P2** → 见下节（已完成）。

---

### Run 20 路线 B：P2 全身惯性对齐 USD — **spawn 验收部分 PASS（2026-06-16）**

**目的：** 对齐 URDF 与 USD 刚体参数（原拟 P2 只查 `torso_link`）。

**发现：** `torso_link` **本就一致**（mass=17.789）；真正差异在 **腿+骨盆+臂**（URDF 总质量 **59.3 kg** vs USD **51.4 kg**）。

**改动：** 从 `h1_minimal.usd` 导出 PhysX 参数，写入 `h1_minimal.urdf` 全部 **20 link** 的 `<inertial>`（`apply_h1_usd_inertia_to_urdf.py`）。

**验收（`compare_h1_body_masses.py`）：** 各 link 质量 delta **全 0.000**；总质量 **51.437 kg**（= USD）。

**验收（`compare_h1_spawn.py`，P1+P2 后）：**

| step | 指标 | P2 前 URDF | **P2 后 URDF** | USD |
| --- | --- | --- | --- | --- |
| 50 | 脚 z | 0.073 | 0.073 | 0.073 |
| 100 | `pitch` | -0.355 | **-0.272** | 0.000 |
| 200 | `pitch` | -0.132 | -0.195 | -0.105 |

**结论：** P2 **刚体对齐 PASS**；`pitch` 有改善但未完全对齐 → **须 Run 23 训+play** 验步态。

---

### Run 23：P0+P1+P2 后 fresh 训（待启动）

**目的：** 全身惯性对齐 USD 后 fresh 训；奖励仍 Flat-10；**勿 resume** Run 20～22。

**Asset：** P0 删传感器 + P1 脚碰撞 + P2 全身惯性对齐 USD。

**训练命令：**

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Flat-Unitree-My_H1 \
  --headless --max_iterations=35000
```

**验收：** ~**1000 iter** play；TB `feet_air` 目标 **~0.05**；对称/小抬腿。

**现象 / 结论：** *（待训）*

---

### Flat Run 1：专用 Flat task + plane 训（`2026-06-12_13-04-07`）

**目的：** **train=play=plane**，先跑通平地再走 Rough；修复原 `flat_env_cfg`（误拷 G1）为继承 `UnitreeMyH1RoughEnvCfg`。

**改动（`flat_env_cfg.py` + 新 task）：**

| 类别 | 内容 |
| --- | --- |
| task | **`RobotLab-Isaac-Velocity-Flat-Unitree-My_H1`** |
| 日志 | **`unitree_my_h1_flat/`** |
| 地形 | **plane**（无 height_scan） |
| 奖励相对 Rough | `variance=-0.5`；关 `feet_height` / `feet_height_body`（Run 18 线） |
| 命令 | 继承 rough：`lin_vel_x (0,0.5)`，`y/z=0` |

**现象（~3500 iter）：**

- `feet_air` smoothed **~0.029**；`track` smoothed **~2.78**；`variance` **~-0.012**。

**验收结论（play，`--num_envs 1 --forward`）：**

- task **`...Flat-My_H1`** + ckpt **`unitree_my_h1_flat/2026-06-12_13-04-07/model_3499.pt`**。
- step 1：`cmd=[0.3,0,0]`；step 100：`vx≈0.31`，`pos_x` 增加，**左脚着地、右脚离地**。
- **训练 PASS**；**步态部分 PASS**（能跟速、能前走、有单脚支撑）。
- **仍 FAIL（拟人/抬脚）：** 抬脚低（~3–5 cm）、膝不够弯；因 `feet_air` 只奖时间、`joint_pos_penalty=-1.0` 拉腿回站姿、未开 `feet_height_body`。

### Flat-2：抬脚高度 + 减站姿惩罚（`2026-06-12_14-22-42`）

**目的：** 在 Flat-1 上专治 **抬脚低、膝难弯**；仅改 `flat_env_cfg.py`，Rough 不动。

**改动（相对 Flat-1）：**

| 项 | Flat-1 | Flat-2（训时 cfg） |
| --- | --- | --- |
| `joint_pos_penalty` | `-1.0` | **`-0.3`**（仍含肩肘 `.*`） |
| `feet_air_time` | `2.0` | **`2.5`**（在 2.0 上再加强「脚离地时间」；**非官方值，Flat-2 试错**） |
| `feet_height_body` | `0` | **`-1.0`**，`target_height=-0.3` |
| `feet_height` | `0` | `0` |
| `joint_deviation_arms` | `-0.2`（继承） | **未改** |
| `feet_air_time_variance` | `-0.5` | `-0.5` |

**`feet_air_time=2.5` 说明：** 官方 H1 **1.0**；My_H1 rough Run 18 **2.0**。Flat-2 因 Flat-1 抬脚太低，在 **2.0 上再加 0.5** 与 `feet_height_body` 叠加强抬脚；结果 **抬过头**（见下），Flat-3 已改回 **2.0**。

**现象（~21250 iter）：**

- `feet_air` smoothed **~0.31**（官方 Flat **~0.05** 量级，**明显过高**）。
- `track` smoothed **~2.76**；`variance` **~-0.026**。

**验收结论（play，`--num_envs 1 --forward`）：**

- ckpt：**`unitree_my_h1_flat/2026-06-12_14-22-42/model_21250.pt`**（或中期 ~21k）。
- **抬脚 PASS**：如 step 100 `right_foot_z≈0.20`、左脚贴地 `≈0.07`，腿能明显抬起。
- **跟速/能走 PASS**：`vx≈0.25–0.37`，有位移。
- **步态自然 FAIL**：**踩高跷感** — 一脚高举、一脚钉地，髋膝不协调；`feet_height_body=-1` + `feet_air=2.5` + `target=-0.3` 叠加强抬脚。
- **摆臂 FAIL**：手臂下垂不动；`joint_deviation_arms=-0.2` + `joint_pos_penalty` 仍罚肩肘偏离默认，**无摆臂奖励**（velocity PPO 常态）。

### Flat-3：治高跷（`2026-06-12_23-28-20`）

**目的：** 保留 Flat-2 抬脚能力，减弱「高跷感」。

**改动（相对 Flat-2）：**

| 项 | Flat-2 | Flat-3 |
| --- | --- | --- |
| `feet_air_time` | `2.5` | **`2.0`** |
| `feet_height_body` | `-1.0`，`target=-0.3` | **`-0.5`**，`target=-0.2` |
| 其余 | — | `joint_pos_penalty=-0.3`，`feet_air_time_variance=-0.5` |
| 训练命令 | — | `lin_vel_x (0, 0.5)`（继承 rough） |

**现象（~16200 iter，训至约 46% / `max_iterations=35000`）：**

- `feet_air` smoothed **~0.073**，仍上涨；介于 Flat-1（~0.029）与 Flat-2（~0.31）之间。
- `track_lin_vel_xy_exp` smoothed **~2.80**，~5k 后平台。
- `feet_air_time_variance` **~-0.019**，更负 → 左右脚离地时间不对称在增大。

![TensorBoard：Flat-3 feet_air / track / variance（~16201 step）](./images/flat3_tensorboard_feet_air_track.png)

**验收结论（play，`--num_envs 1 --forward`，ckpt `model_16200.pt`～`16250.pt`）：**

| 速度 | 结论 | 要点 |
| --- | --- | --- |
| **`--forward_vel 0.3`** | **训练 PASS / 步态部分 PASS** | `vx≈0.31` 跟速；400 step `pos_x≈1.2 m`；左右 **交替** 触地/离地；`fallen=False`；比 Flat-2 自然 |
| **`--forward_vel 0.5`** | **跟速 PASS / 对称 FAIL** | 能走、不摔；**仍瘸** — 右脚抬更高、着地姿态与左脚不一致 |
| **`--forward_vel 1.0`** | **跟速 PASS / 步态 FAIL** | `vx≈1.0` 可跟上，但 **非跑**（无双脚腾空）；step 200 `left_foot_z≈0.073`、`right_foot_z≈0.264`（右脚 ~3.6×）；右前掌 / 左后跟着地感；**训练 OOD**（训时上限 0.5） |

![play：Flat-3 @0.3 m/s 能走、交替迈步](./images/flat3_play_forward03_ok.png)

![play：Flat-3 @1.0 m/s debug 左右脚高度不对称](./images/flat3_play_forward10_limp_debug.png)

**小结：**

- Flat-3 是 **My_H1 Flat 当前最好主线**（0.3 m/s 下能走、比 Flat-2 自然）。
- **拟人 / 对称 FAIL**：0.5 也瘸；`variance=-0.5` 偏弱 + `joint_mirror=0`；1.0 为分布外速度。
- **摆臂**：仍 FAIL（与官方一致，reward 罚肩肘偏离默认）。
- **Flat-4 方向：** 见下节 **Flat-4**（已训已 play）。

### Flat-4：高速 + 对称（`2026-06-13_04-49-34`）

**目的：** 在 Flat-3 基础上 **(1) 训到 1.0 m/s 快步 (2) 减轻左右瘸态**。

**改动（相对 Flat-3）：**

| 项 | Flat-3 | Flat-4 |
| --- | --- | --- |
| `lin_vel_x` | `(0, 0.5)` | **`(0, 1.0)`** |
| `command_levels_lin_vel` | 关 | **开**，倍率 `(0.2, 1.0)` 渐进 |
| `feet_air_time_variance` | `-0.5` | **`-1.0`** |
| `feet_contact` | `0` | **`-0.25`**，`expect_contact_num=1` |
| `feet_air_time.threshold` | `0.6` | **`0.4`** |
| `joint_mirror` | `0` | **0**（Run 17 已证有害） |
| 训练 | Flat-3 run | **fresh** 新训 |
| `max_iterations` | 35000 | **35000** |

**说明（`feet_contact` / `expect_contact_num`）：** 基类默认 `expect_contact_num=2`（双脚着地）；Flat-4 显式改为 **1**（移动时鼓励单脚支撑）。Flat-3 该项 weight=0 未启用。

**现象（~27450 iter，约 8h）：**

- `feet_air` smoothed **~0.23**（Flat-3 末期 **~0.07**；**高 ≠ 更自然**，可能一条腿大摆、另一条短摆/蹭地）。
- `track_lin_vel_xy_exp` smoothed **~2.71**。
- `feet_air_time_variance` **~-0.037**（仍负 → 左右相位/时长仍不平衡）。
- 训练早期（~74 iter）`feet_air=0` 属正常：策略尚未进入单脚支撑相位；~3k step 后明显上涨。

![TensorBoard：Flat-4 feet_air / track / variance（~27408 step）](./images/flat4_tensorboard_feet_air_track.png)

**验收结论（play，`--num_envs 1 --forward`，ckpt `model_27450.pt` 附近）：**

| 速度 | 跟速 | 对称 / 步态 | 要点 |
| --- | --- | --- | --- |
| **`--forward_vel 0.5`** | **PASS** `vx≈0.48` | **FAIL** | step 100：`left_foot_z≈0.243`、`right_foot_z≈0.078`（左 **~3×** 高）；左 F / 右 T；**左右摆幅差大，看着仍不对** |
| **`--forward_vel 1.0`** | **PASS** | **FAIL** | 能跟；仍不对称（与 Flat-3 类似，主导腿可能左/右轮换） |
| **`--forward_vel 2.0`** | 可跟 `vx≈1.7~1.9` | **FAIL** | **OOD**（训上限 1.0）；step 200 右高左低，step 300 双 contact、左踝仍高 → **拖/蹭地感**；**不作主验收** |

![play：Flat-4 @0.5 m/s 左踝高、右贴地](./images/flat4_play_forward05_left_high.png)

![play：Flat-4 @2.0 m/s（OOD）不对称 + debug](./images/flat4_play_forward20_asymmetric.png)

**小结：**

- **跟速 PASS**（0.5 / 1.0 训练范围内）。
- **迈步分项 PASS**（TB `feet_air≈0.23`）。
- **对称 / 自然 FAIL**：0.5 下 **左腿常抬更高**，但与 Flat-3 同为 **左右摆幅不一致**；`variance=-1.0` + `feet_contact` 未根治。
- **摆臂**：仍 FAIL（未改 reward）。
- **Flat-5 方向：** 见下节 **Flat-5**（已训已 play @13k）。

### Flat-5：加强对称惩罚 + 蹭地罚（`2026-06-13_12-59-13` → resume `2026-06-13_17-24-46`）

**目的：** 在 Flat-4 基础上治 **左右摆幅不一致**（一条腿大摆、另一条贴地/蹭地）；Flat-4 训到 ~27k `variance` 已平台，改 cfg **fresh + resume** 继续。

**改动（相对 Flat-4）：**

| 项 | Flat-4 | Flat-5 |
| --- | --- | --- |
| `feet_air_time_variance` | `-1.0` | **`-2.0`** |
| `feet_height_body` | `-0.5`（flat） | **`-0.35`** |
| `feet_slide` | `-0.2`（rough 默认） | **`-0.35`**（`rough_env_cfg.py`） |
| 其余 | — | `feet_air=2.0`、`feet_contact=-0.25` + **`expect_contact_num=1`**、`threshold=0.4`、速度 curriculum **同 Flat-4** |
| 训练 | Flat-4 run | **fresh** `12-59-13` → **`--resume`** 自 `model_3500.pt` → 新目录 **`17-24-46`** |

**说明（resume）：** train 的 `--checkpoint` 只能写 **文件名**（如 `model_3500.pt`），勿写完整路径；play 的 `--checkpoint` 可写完整路径。resume 后 TB / ckpt 看 **新目录** `17-24-46`，不是 `12-59-13`。

**现象（~13187 iter，resume 合计 ~13k）：**

- `feet_air` smoothed **~0.23**（与 Flat-4 末期 **~0.23** 相当 → 迈步分项仍 PASS）。
- `track_lin_vel_xy_exp` smoothed **~2.71**（跟速 PASS）。
- `feet_air_time_variance` **~-0.066**（Flat-4 末期 **~-0.037** → **更负、更差**；加强 `-2.0` 未改善对称）。
- 早期 ~3500 step play：跟速 OK，双足蹭地、步态乱（正常，未收敛）。

**验收结论（play，`--num_envs 1 --forward`，ckpt **`model_13150.pt`**，`17-24-46`）：**

| 速度 | 跟速 | 对称 / 步态 | 要点 |
| --- | --- | --- | --- |
| **`--forward_vel 0.5`** | **PASS** `vx≈0.49~0.52` | **FAIL** | step 200–300：左 **~0.070**、右 **~0.105–0.151**；左长期 **贴地当支撑/拖地带** |
| **`--forward_vel 1.0`** | **PASS** `vx≈1.0` | **FAIL** | step 100：左 **~0.070** vs 右 **~0.234**（差 **~0.16**）；不对称被高速放大 |

规律：**左脚常在 ~0.07 m（贴地桩）**，**右脚负责大摆（0.15–0.23 m）**；与 Flat-4 @0.5「左高右低」可能 **相位不同**，本质同为 **一条腿低、一条腿高**。摆动相 `foot_z` 峰值差 **> ~0.08** → 对称粗 **FAIL**。

![play：Flat-5 @0.5 m/s 左贴地、跟速 OK + debug](./images/flat5_play_forward05_left_low_debug.png)

![play：Flat-5 @1.0 m/s 左 ~0.07、右大摆 + debug](./images/flat5_play_forward10_left_low_debug.png)

**小结：**

- **跟速 PASS**（0.5 / 1.0）。
- **迈步分项 PASS**（TB `feet_air≈0.22`）。
- **对称 FAIL**：`variance` **-1.0 → -2.0** 后 TB **更负**，play **未变好**；策略仍用「一脚钉地 + 一脚大摆」凑 `feet_air`。
- **摆臂**：仍 FAIL。
- **是否续训：** 可训到 **20k～25k 再 play**；若 `variance` 仍 **< -0.04** 且 play 仍左 ~0.07 / 右 ~0.15+ → **开 Flat-6 改 cfg**，勿只加训。
- **Flat-6 方向：** 见下节 **Flat-6**（已训已 play）。

### Flat-6：降 `feet_air` + 加强 slide（`2026-06-13_20-25-29`）

**目的：** Flat-5 @13k 仍「左脚贴地 + 右脚大摆」；减弱 `feet_air` 对「一脚狂摆」的激励，flat 覆盖 **`feet_slide=-0.5`**。

**改动（相对 Flat-5）：**

| 项 | Flat-5 | Flat-6 |
| --- | --- | --- |
| `feet_air_time` | `2.0` | **`1.5`** |
| `feet_slide`（flat 覆盖） | 继承 `-0.35` | **`-0.5`** |
| 其余 | — | `variance=-2.0`、`feet_contact` + **`expect=1`**、`threshold=0.4`、速度 curriculum **同 Flat-4/5** |
| 训练 | resume Flat-5 | **fresh** |

**现象（~24869 iter）：**

- `feet_air` smoothed **~0.121**（略低于 Flat-5 @13k **~0.23**，Flat-7 末期 **~0.133**）。
- `track_lin_vel_xy_exp` smoothed **~2.74**。
- `feet_air_time_variance` **~-0.051**（Flat-5 **~-0.066** 略好，仍负）。

**验收结论（play @0.5/1.0，ckpt **`model_24850.pt`** 附近）：**

- **跟速 PASS**。
- **对称 FAIL**：与 Flat-5 **同型**（一脚 **~0.07** 贴地、另一脚 **0.13～0.23** 大摆）；`feet_slide` 加重 **未根治**。
- **Flat-7 方向：** 见下节。

### Flat-7：再加强 `feet_slide`（`2026-06-14_03-43-18`）

**目的：** Flat-6 @25k 仍不对称；`feet_slide` **-0.5 → -0.65**（flat 覆盖）。

**改动（相对 Flat-6）：**

| 项 | Flat-6 | Flat-7 |
| --- | --- | --- |
| `feet_slide`（flat） | `-0.5` | **`-0.65`** |
| 其余 | — | `feet_air=1.5`、`variance=-2.0` 等 **不变** |

**现象（~29675 iter）：**

- `feet_air` smoothed **~0.133**。
- `track` smoothed **~2.73**。
- `variance` **~-0.054**（比 Flat-6 **略更负**）。

> **注：** Flat-7 TB 截图当时未单独归档；下列数值来自 TensorBoard 读数，非截图。

**验收结论（play，`model_29650.pt`，`--forward`）：**

| 速度 | 跟速 | 对称 / 步态 | 要点 |
| --- | --- | --- | --- |
| **`--forward_vel 0.5`** | **PASS** | **FAIL** | step 100：左 **~0.165** / 右 **~0.070**；step 200：左 **~0.070** / 右 **~0.131** → **相位不同但摆幅差仍大** |
| **`--forward_vel 1.0`** | **PASS** | **FAIL** | 左 **~0.07** vs 右 **~0.15+** 模式仍在 |

- **续训到 35k 性价比低**（Flat-4 已证 variance 平台）；**Flat-8 改 cfg**。

### Flat-8：官方 gait 数字 + 保留 curriculum（`2026-06-14_15-46-31`）

**目的：** My_H1 Flat **从未** 用 Isaac **`feet_air=1.0/0.6`** 等温和配方；在保留 **只前进 + 速度 curriculum** 下 **fresh** 试官方数字（仍用 robot_lab **track=3.0** 等大 weight，非 Isaac 全量对齐）。

**改动（相对 Flat-7）：**

| 项 | Flat-7 | Flat-8 |
| --- | --- | --- |
| `feet_air_time` | `1.5` / `0.4` | **`1.0` / `0.6`** |
| `feet_slide` | `-0.65` | **`-0.2`**（flat） |
| `feet_air_time_variance` | `-2.0` | **`-1.0`** |
| `feet_contact` | `-0.25`, expect=1 | **0** |
| `feet_height_body` | `-0.35` | **0** |
| `joint_pos_penalty` | `-0.3` | **`-1.0`** |
| curriculum | 开 | **开**（保留） |

**现象（~13835 iter，约 40% 目标 35k 时停训）：**

- `feet_air` **~0.035**（接近官方 Flat baseline **~0.05** 量级，远低于 Flat-7 **~0.13**）。
- `variance` **~-0.019**（**最好**的一轮）。
- `track` **~2.75**。

![TensorBoard：Flat-8 feet_air / track / variance（~13718 step）](./images/flat8_tensorboard_feet_air_track.png)

**验收结论（play @0.5，`model_13700.pt`）：**

- **跟速 PASS**（`vx≈0.52`）。
- step 300：左 **~0.070** / 右 **~0.084** → **高度差小**，但 **两脚都贴地 ~0.07–0.08**，**拖步/碎步感**（别扭类型从「大摆瘸」变为 **抬脚不足**）。
- **对称粗边界**（差 ~0.01–0.06），**自然 / 抬脚 FAIL**。
- **13k 偏早**；结论：**官方数字 + curriculum 减轻瘸态，但未达 Isaac 预训练观感**。

![play：Flat-8 @0.5 m/s 两脚贴地、跟速 OK + debug](./images/flat8_play_forward05_shuffle_debug.png)

**小结：** 用户倾向 **Isaac 预训练步态** → 见下节 **Isaac 对齐线（Flat-9）**。

### Isaac 对齐线 Flat-9（reward 数字对齐 Isaac；底座仍 robot_lab）

**目的：** 将 **`rough_env_cfg` + `flat_env_cfg` + Flat PPO** 的 **reward 数字** 对齐 **Isaac Lab `H1RoughEnvCfg` / `H1FlatEnvCfg` / `H1FlatPPORunnerCfg`**（非仅 reward 数字 halfway 的 Flat-8）。**Events/actions/commands 仍为 robot_lab 默认** → 见 Flat-10。

**改动（相对 Flat-8 / 旧 My_H1 线）：**

| 类别 | Flat-8 等旧线 | **Flat-9（当前磁盘 cfg）** |
| --- | --- | --- |
| `track` weight | `3.0` | **`1.0`** |
| Rough `feet_air` | `2.0` / `0.6` | **`0.25` / `0.4`**；Flat 覆盖 **`1.0` / `0.6`** |
| `feet_slide` | `-0.2～-0.65` | **`-0.25`**（rough）；Flat **继承** |
| `variance` / `feet_contact` / `feet_height_body` | 各种启用 | **全 0** |
| `joint_pos_penalty` | `-0.3～-1.0` | **0**；用 **joint_deviation + 踝 limit -1.0** |
| `flat_orientation` | `-0.2` | **`-1.0`** |
| `base_lin_vel` obs | **None** | **恢复（有）** |
| 命令 | 只前进 | **`ang_vel_z∈(-1,1)` 可转向**（Isaac rough） |
| 速度 curriculum | Flat-4 起 **开** | **关**（Isaac 无） |
| Flat PPO 网络 | `[512,256,128]`，35000 iter | **`[128,128,128]`，`1000` iter** |

**训练 / 验收：** **须 fresh**；**勿 resume** Flat-8 ckpt（**网络结构不同**）。训完 play 仍用 **`--num_envs 1 --forward --follow`** @0.5 / 1.0。

#### Flat-9 训练记录（`2026-06-15_09-08-40` → resume `2026-06-15_09-59-37`）

| 项 | 内容 |
| --- | --- |
| 训 | fresh `09-08-40`；后 **`--resume --load_run 09-08-40`** → 新日志 **`09-59-37`** |
| 预算 | 用户设 **`--max_iterations 35000`**（远超 Isaac Flat 默认 1000） |
| ckpt 主验 | **`09-59-37` / `model_13550.pt`**（~13.5k iter） |

**现象（TB @~13.5k，`09-59-37`）：**

- `feet_air` smoothed **~0.22**（仍上升；3k 时曾 **~0.014**）。
- `track_lin_vel_xy_exp` smoothed **~0.93**（weight=1.0）。

![TensorBoard：Flat-9 fresh @~2999（`09-08-40`，用户截图）](./images/flat9_tensorboard_2999.png)

![TensorBoard：Flat-9 resume @~13555（`09-59-37`，用户截图）](./images/flat9_tensorboard_13555.png)

**验收结论（play @0.5，`model_13550.pt`）：**

- **跟速 PASS**（`vx≈0.5`）。
- **对称粗 PASS**：左右 **交替** 抬脚（非 Flat-4～7 的「一脚永远钉地」）。
- **抬脚过高 FAIL**：摆动脚 **z≈0.30–0.40**，支撑脚 **~0.07**（~30 cm 级高抬腿）。
- **更早 ckpt**（3k～8k）play **同样偏高** → 非单纯「训太久」，**MDP 底座仍与 Isaac 不同**（action 0.25、全套 DR、Threshold 命令等）。

![play：Flat-9 `model_13550` @0.5 — 对称但抬过高 + debug（用户截图）](./images/flat9_play_13550_high_lift.png)

**小结：** reward 数字对齐 **必要但不充分** → 触发 **Flat-10 底座对齐**。

---

### Flat-10：MDP 底座对齐 Isaac（`2026-06-15_13-15-36`）— **当前达标线**

**目的：** 在 Flat-9 reward 数字基础上，把 **events / actions / commands / 惩罚项 / 地形** 等与 Isaac `H1RoughEnvCfg` 对齐。

**改动（相对 Flat-9）：**

| 类别 | Flat-9 | **Flat-10** |
| --- | --- | --- |
| `joint_pos.scale` | **0.25** | **0.5**（Isaac 默认） |
| Commands | `UniformThresholdVelocityCommand` | **`UniformVelocityCommand`** |
| 域随机化 | robot_lab 全套（push/质量/COM/执行器增益…） | **按 Isaac 关/固定**（push=None，摩擦 0.8/0.6，reset 初速全 0…） |
| `ang_vel_xy_l2` | 0 | **-0.05** |
| 地形 restitution | 1.0 | **0.0** |
| Obs scale | 自定义 2.0/0.25 等 | **移除**（Isaac 默认 1.0） |
| Reward 数字 | 同 Isaac | **同 Flat-9**（Flat：`feet_air=1.0/0.6`） |

**训练：**

| 项 | 内容 |
| --- | --- |
| Run | **`2026-06-15_13-15-36`** |
| 命令 | `train.py --task RobotLab-Isaac-Velocity-Flat-Unitree-My_H1 --headless --max_iterations 35000` |
| 须 **fresh** | **勿 resume** Flat-9 ckpt |

**现象（TB）：**

| iter | `feet_air` | `track` | 解读 |
| --- | --- | --- | --- |
| ~1000 | **~0.05** | **~0.97** 平台 | **sweet spot** |
| ~2500 | 上升 | ~0.97 | track 仍高 |
| ~4000 | **~0.18** | **~0.94↓** 波动 | 过训：feet_air 饱和、track 掉 |

![TensorBoard：Flat-10 @~4000（`13-15-36`，用户截图）](./images/flat10_tensorboard_4000.png)

**验收结论（play，`model_1000.pt`，@0.5，`--forward --follow`）：— 用户认定达标**

- **跟速 PASS**：`vx≈0.45`。
- **对称 PASS**：step 300 左着地/右抬，step 400 右着地/左抬。
- **抬脚 PASS（小抬腿）**：摆动脚 **z≈0.12**，支撑 **~0.07**，离地 **~5 cm**。
- **躯干稳**：`roll/pitch≈0`。
- **推荐主 ckpt：** **`model_1000.pt`**（Isaac Flat 官方预算）；**勿用 `model_4000.pt`**（左腿后翘、右腿蹭地，track 已掉）。

![play：Flat-10 `model_1000` @0.5 — 小抬腿 PASS + debug（用户截图）](./images/flat10_play_1000_pass.png)

**对照（play `model_4000.pt` @0.5）— FAIL：**

- 左腿单腿后翘、右腿蹭地；`feet_air` 过饱和。

![play：Flat-10 `model_4000` @0.5 — 后翘瘸步 FAIL（用户截图）](./images/flat10_play_4000_fail.png)

**Play 命令（归档）：**

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Velocity-Flat-Unitree-My_H1 \
  --load_run 2026-06-15_13-15-36 \
  --checkpoint logs/rsl_rl/unitree_my_h1_flat/2026-06-15_13-15-36/model_1000.pt \
  --num_envs 1 --forward --forward_vel 0.5 --follow --hold_seconds 0
```

**play 视口注意：** 勿点 Isaac UI **Pause**（与 Python 循环不同步，画面会定住但终端仍刷）；用 **`--follow`** 防机器人走出定焦镜头。

---

### Flat 线对照：为何早期「能跑但双腿不平衡」？

> **机制：** `feet_air_time` 奖 **单脚支撑时长**（上限 0.6s），**不奖对称、不罚摆高**。捷径 = **一脚蹭地跟速 + 另一脚大摆凑 air time**。

| 线 | 典型 play @0.5 | 跟速 | 对称 | 抬脚 | 主因 |
| --- | --- | --- | --- | --- | --- |
| **Flat-4～7** | 一脚 **~0.07** 贴地，另一脚 **0.15～0.40** | PASS | **FAIL** | 过高 | 大 `feet_air` + `variance` 补丁 + `track=3.0` |
| **Flat-8** | 两脚都 **~0.07–0.08** 拖步 | PASS | 粗边界 | 过低 | 官方数字 halfway + curriculum |
| **Flat-9 @13k** | 交替但 **z≈0.30–0.40** | PASS | **粗 PASS** | **过高** | reward 对齐但 **底座未对齐** + 训过久 |
| **Flat-10 @1000** | 交替 **z≈0.12** / **~0.07** | PASS | **PASS** | **小抬腿 PASS** | **底座对齐** + **~1000 iter** |

**与 Nucleus 预训练：** Nucleus 为 **Isaac Rough 训满**；Flat-10 为 **robot_lab Flat @1000**。观感已接近用户目标；Nucleus 仍可作为 **1.0 m/s Rough** 上限对照。

---

**改动：**

| 类别 | 内容 |
| --- | --- |
| 任务 | 官方 H1 Flat，非 My_H1 |
| 用途 | 作为脚离地 / 步态参考 |
| ckpt | `unitree_h1_flat/2026-06-08_10-20-32/model_14999.pt` |

**现象：**

- `feet_air≈0.05+`，明显高于 My_H1 当前结果。

![TensorBoard：官方 H1 Flat baseline](./images/official_h1_flat_tensorboard.png)

**验收结论：**

- 作为对照：My_H1 当前脚离地分项远低于官方 Flat。

### 官方 H1 Rough baseline（robot_lab 自训）

**改动：**

| 类别 | 内容 |
| --- | --- |
| 任务 | 官方 H1 Rough，非 My_H1 |
| 用途 | 与 My_H1 Rough 系列对照 |
| ckpt | `unitree_h1_rough/2026-06-08_14-13-36/model_15000.pt` |

**现象：**

- `feet_air≈0.016`，track 约 ~1.44。

![TensorBoard：官方 H1 Rough baseline](./images/official_h1_rough_tensorboard.png)

![play：官方 H1 Rough 叉腿侧挪（Run 对照）](./images/official_h1_rough_play_01.png)

![play：官方 H1 Rough 小跳感（Run 对照）](./images/official_h1_rough_play_02.png)

**验收结论：**

- play 仍是叉腿侧挪 / 小跳，未通过标准双足走。
- 曲线过线也必须做 play 验收。
- **注意：** 此为 **robot_lab 官方 task + 本机自训 ckpt**，**不是** Isaac Lab Nucleus 预训练（见下节）。

---

### Isaac Lab 官方 H1 预训练 play（Nucleus，对照上限）

> **与上两节区分：** 本节是 **Isaac Lab 仓库 + Nucleus 预训练权重**；上两节是 **robot_lab 官方 cfg + 本机 `logs/rsl_rl/unitree_h1_*` 自训**。

**入口：**

| 项 | 内容 |
| --- | --- |
| 仓库 | `~/project/IsaacLab`（非 robot_lab） |
| task | `Isaac-Velocity-Rough-H1-Play-v0`（Flat 可用 `Isaac-Velocity-Flat-H1-Play-v0`） |
| 权重 | Nucleus `PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Rough-H1-v0/checkpoint.pt` |
| Play 命令速度 | Play cfg 内 **`lin_vel_x=(1.0, 1.0)`** m/s（非 robot_lab 的 `--forward 0.3`） |

**加载坑（rsl_rl 5.3）：**

- 直接 `--use_pretrained_checkpoint` 会 **`KeyError: 'actor_state_dict'`**（Nucleus ckpt 为旧格式 `model_state_dict`）。
- **处理：** 将 ckpt 转为 `actor_state_dict` / `critic_state_dict`（`actor.*`→`mlp.*`，`std`→`distribution.std_param`）后，用 `--checkpoint .../checkpoint_rslrl5.pt` play。

**现象（Rough 预训练，视口 + 交互）：**

- **交替抬腿 PASS**：左右脚有明确离地/触地相位。
- **跟速 PASS**：Play 模式恒 **1.0 m/s**，步幅大、观感比 Flat-3 @0.3 更「灵活」。
- **非跑 PASS 边界**：仍是 **velocity 快步走**，无双脚腾空真跑。
- **摆臂 FAIL**：`joint_deviation_arms=-0.2` 罚肩肘偏离默认，**手臂下垂不摆**（与 robot_lab 同理，非 bug）。
- **左右对称**：目视 **优于 Flat-3 @0.5/1.0 的瘸态**；未逐帧量化 left/right foot_z。

![play：Isaac Lab Nucleus 预训练 H1 Rough @1.0 m/s](./images/isaaclab_official_h1_pretrained_play.png)

**与 Flat-3 对比要点：**

| 维度 | Isaac Lab 预训练 | Flat-3（My_H1） |
| --- | --- | --- |
| 速度 | play **1.0 m/s**（且训过 0～1.0） | 训 **0～0.5**，play 常用 0.3 |
| 抬脚视觉 | 快步下自然更高 | 0.3 偏低；1.0 OOD 时右脚过高 |
| 对称性 | 目视较对称 | **0.5 / 1.0 均瘸**（右脚更高） |
| 摆臂 | 不摆 | 不摆 |
| 权重来源 | Nucleus 下载 | 本机 `unitree_my_h1_flat/2026-06-12_23-28-20` |

**验收结论：**

- 作为 **「同一 H1 USD、velocity 任务、接近上限」** 的参考：**腿 / 跟速 / 交替迈步** 优于当前 Flat-3。
- **不能**直接当 My_H1 的 ckpt（task / obs / cfg 不同）；只作 **步态风格与速度对照**。

#### Isaac Lab 官方 H1 自训（`2026-06-15_03-26-47`，Rough，对照实验）

> **与 Nucleus 预训练对比：** 同 **Isaac Lab 仓库 + 同 Rough task 族**；差别是 **本机从零训 ~3000 iter** vs **Nucleus 训满发布**。

| 项 | 内容 |
| --- | --- |
| 仓库 | `~/project/IsaacLab` |
| task 训 | `Isaac-Velocity-Rough-H1-v0` |
| task play | `Isaac-Velocity-Rough-H1-Play-v0` |
| 日志 | `IsaacLab/logs/rsl_rl/h1_rough/2026-06-15_03-26-47/` |
| ckpt | **`model_2999.pt`**（训满默认 3000 iter） |

**现象（~2999 iter）：**

- `feet_air` smoothed **~0.014**（Nucleus 预训练 **明显更高**；迈步分项 **远未饱和**）。
- `track_lin_vel_xy_exp` smoothed **~0.93**（weight=**1.0**，**不可与** My_H1 `track≈2.7` **直接比绝对值**）。

![TensorBoard：Isaac 自训 h1_rough @2999（用户截图，`03-26-47`）](./images/isaac_h1_selftrain_rough_tensorboard_2999.png)

**验收结论（play，`model_2999.pt`）：**

- **跟速**：Play cfg **~1.0 m/s**，策略 **能跟但步态未成型**。
- **对称 / 自然**：用户 play 截图（Rough 地形、速度命令箭头可见）：**一前一后、左右不像交替走**；用户反馈 **「是要对称点」** → **对称 FAIL**。

![play：Isaac 自训 h1_rough `model_2999.pt`（用户截图）](./images/isaac_h1_selftrain_rough_play_2999.png)

- **解读：** **3k iter 不够**；不等于 Isaac 代码训不出对称。续训 **10k** 或对照 Nucleus 才公平。
- **对 My_H1 启示：** Isaac **轻 reward + 训够** 才可能接近预训练；My_H1 Flat-8 halfway 官方数字 **≠ 全面对齐 Isaac** → 先 **Flat-9 reward 对齐**，再 **Flat-10 MDP 底座对齐**（当前 PASS）。

---

### 附录：环境 / 调试截图

#### 环境验证 / spawn（第 5 章相关）

![zero_agent：初次 spawn 姿态检查](./images/ch5_zero_agent_spawn_check.png)

![Isaac 视口：H1 spawn 早期](./images/ch5_isaac_view_01.png)

![Isaac 视口：关节/坐标轴调试](./images/ch5_isaac_view_02.png)

![Isaac 视口：spawn 姿态](./images/ch5_isaac_view_03.png)

![Isaac 视口：H1 站立/选中](./images/ch5_isaac_view_04.png)

![Isaac 视口：Translate Z 检查](./images/ch5_isaac_view_05.png)

![Property：Robot Translate Z≈1.05，spawn 高度正常](./images/ch5_spawn_height_z105.png)

![spawn 瞬间：关节扭麻花（改 joint_pos 前）](./images/ch5_spawn_twisted_pose.png)

![spawn 瞬间：双足着地、大致站立（改 joint_pos 后）](./images/ch5_spawn_standing_ok.png)

![Isaac 视口（早期环境验证）](./images/ch5_isaac_view_extra_01.png)

![Isaac 视口（早期环境验证）](./images/ch5_isaac_view_extra_02.png)

![Isaac 视口（早期环境验证）](./images/ch5_isaac_view_extra_03.png)

![Isaac 视口（早期环境验证）](./images/ch5_isaac_view_extra_04.png)

#### 工具 / 终端 / Isaac 调试

![终端无输出（viewer 模式）](./images/appendix_terminal_no_output.png)

![Cursor Automations 弹窗（无关训练）](./images/appendix_cursor_automation_popup.png)

![Isaac Move 工具抢键盘（非 play 控速）](./images/appendix_isaac_move_tool.png)

![TensorBoard 搜索字段](./images/appendix_tensorboard_search.png)

![终端括号粘贴模式乱码](./images/appendix_terminal_bracketed_paste.png)

![自动化误触发说明](./images/appendix_automation_confusion.png)

![杂项：GitHub PAT 创建页（非训练截图，归档备查）](./images/appendix_misc_01.png)

### 当前最值得复验

- **★ Flat-10 PASS（主 ckpt）：** `2026-06-15_13-15-36` / **`model_1000.pt`**；小抬腿、左右交替 @0.5；补验 @1.0。
- **历史对照：** Flat-4～9 见第 10 章「Flat 线对照」表；勿再 resume 旧线 ckpt 到 Flat-10 cfg。
- **Run 15 R2 站桩 ckpt：** `2026-06-10_10-04-46 / model_3450.pt`（不摔但脚不抬；plane 时代基线）。
- **Run 15 R3～R4 已验 FAIL：** 后跳摔 / 后仰摔。
- **Run 15 R5～R7 已验 FAIL：** 官方奖励 + PD + PPO 对齐；见隔离小结与 zero 对照。
- **官方 H1 baseline play：** 叉腿、右脚蹭地挪（非标准走，但 **优于 My_H1 秒摔**）。
- **Run 16 已验（隔离 PASS / 步态 FAIL）：** `2026-06-12_01-48-27`；USD + 官方化奖励；曲线 `feet_air≈0.015`；play **与官方一样右脚蹭地挪**；**URDF 主因已证实**。
- **Run 17 已验 FAIL：** `2026-06-12_02-39-50`；`joint_mirror=-0.1`；曲线 `feet_air≈0.020`；play 仍蹭地；**mirror 非交替步态杠杆**。
- **Run 18 已验（训练 PASS / 步态部分 PASS）：** `2026-06-12_03-36-58`；Rough；`feet_air≈0.052`；`--plane --forward` play **能前走、微抬脚、仍瘸**；**Rough 未验**。
- **Run 19 已验（Rough FAIL / Plane 部分 PASS）：** `2026-06-12_10-03-07`；`variance=-1` + `feet_height` + 只前进；Rough play **两步卡死**；plane 更对称但更贴地。
- **Flat Run 1 已验：** `2026-06-12_13-04-07`；`feet_air≈0.029`；能走、抬脚低、拟人 FAIL。
- **Flat-2 已验（抬脚 PASS / 自然 FAIL）：** `2026-06-12_14-22-42`；`feet_air≈0.31`；**踩高跷、手臂不摆**；`feet_air=2.5` 试错偏大。
- **Flat-3 已验（0.3 PASS / 对称 FAIL）：** `2026-06-12_23-28-20`；~16200 iter；`feet_air≈0.073`；**0.3 m/s 能走**；0.5 / 1.0 仍瘸；ckpt **`model_16200.pt`～`16250.pt`**。
- **Flat-4 已验（跟速 PASS / 对称 FAIL）：** `2026-06-13_04-49-34`；~27450 iter；`feet_air≈0.23`、`variance≈-0.037`；**0.5/1.0 跟速 OK**；0.5 左 `z≈0.24` vs 右 `≈0.08`；ckpt **`model_27450.pt`**；**2.0 OOD 勿主验**。
- **Isaac Lab Nucleus 预训练 H1：** Rough @**1.0 m/s** 交替迈步、灵活度优于 Flat-3；**不摆臂**；需 **ckpt 格式转换** 才能在 rsl_rl 5.3 play（见第 10 章附录 E）。
- **Flat-5 已验（跟速 PASS / 对称 FAIL）：** resume run **`2026-06-13_17-24-46`**；~**13k** iter；`feet_air≈0.23`、`variance≈**-0.066**`（比 Flat-4 更差）；play @0.5/1.0 **左 ~0.07、右 0.15–0.23**；ckpt **`model_13150.pt`**。
- **Flat-6 已验（跟速 PASS / 对称 FAIL）：** `2026-06-13_20-25-29`；~**25k** iter；`feet_air≈0.12`、`variance≈-0.051`；`feet_air↓`+`slide=-0.5` **未根治**；ckpt **`model_24850.pt`**。
- **Flat-7 已验（跟速 PASS / 对称 FAIL）：** `2026-06-14_03-43-18`；~**30k** iter；`slide=-0.65`；play 相位轮换仍 **摆幅差大**；ckpt **`model_29650.pt`**。
- **Flat-8 已验（跟速 PASS / 贴地别扭）：** `2026-06-14_15-46-31`；~**14k** iter；`feet_air≈0.035`、`variance≈-0.019`；play **两脚 ~0.07–0.08 拖步**；ckpt **`model_13700.pt`**。
- **Flat-9 已验（对称粗 PASS / 抬过高）：** fresh `09-08-40` → resume **`09-59-37`**；~**13.5k**；`feet_air≈0.22`；play 交替但 **z≈0.3–0.4**；ckpt **`model_13550.pt`**；底座未对齐 → Flat-10。
- **Flat-10 已验 PASS（当前目标步态）：** `2026-06-15_13-15-36`；**`model_1000.pt`** @0.5；小抬腿 ~5 cm、左右交替、跟速 OK；**勿用 4k+ ckpt**；见第 10 章 Flat-10 与对照表。
- **Isaac Lab 自训 Rough @3k：** `2026-06-15_03-26-47`；`feet_air≈0.014`；play **远不如 Nucleus**；见第 10 章 Isaac 自训小节。
- **play 要点：** Flat task + flat ckpt；**`--num_envs 1 --forward --follow --hold_seconds 0`**；Flat 不必 `--plane`；验收 **0.5 / 1.0**（勿用 2.0 作主验）；摆动相 `foot_z` 峰值差 **< ~0.08** 为对称粗 PASS；**勿点 Isaac UI Pause**（画面卡死、终端仍刷）。
- **Run 20 已验 FAIL（URDF minimal + Flat-10 奖励）：** `2026-06-16_02-06-53`；`feet_air≈0.14` @1k；play **左脚撑地、右脚常驻空中**；spawn：URDF 更塌/前倾（step100 `pitch≈-0.32`）、脚 z≈0.062；见第 10 章 Run 20。
- **Run 20 路线 B P0 已做：** `h1_minimal.urdf` 删传感器 link；merge 警告消除。
- **Run 21 已验 FAIL（P0 后 fresh）：** `2026-06-16_02-45-30`；@4.5k `feet_air≈0.27`；play **仍单脚抬**。
- **Run 22 已验 FAIL（P0+P1 后 fresh）：** `2026-06-16_04-08-18`；@23k `feet_air≈0.31`；单脚抬；**勿 resume**。
- **P2 已做（全身惯性对齐 USD）：** 20 link；总质量 **51.4 kg**；spawn `pitch@100` **-0.27**；见 P2 节；**Run 23 fresh 训待启动**。
- **若坚持 URDF 路线：** `_USE_OFFICIAL_H1_USD=False`；勿 resume `02-06-53`；用 **`compare_h1_spawn.py`** 验收 spawn；勿堆 reward。
