
# 1. 硬件配置 (来自 3.2.2 节)
NUM_RACKS = 4
SERVERS_PER_RACK = 28
TOTAL_SERVERS = NUM_RACKS * SERVERS_PER_RACK
CPU_PER_SERVER = 26

STATE_DIM = (TOTAL_SERVERS * 3) + 2

# 2. 功耗模型参数 (来自 3.2.2 节)
P_IDLE = 100.0  # W (p_idle)
P_FULL = 300.0  # W (p_full)

# 3. 任务约束 (来自 1.2.1 和 3.2.2 节)
MAX_CPU_PER_SUBTASK = 8


# 5. DRL 奖励函数参数 (*** 关键修改 ***)
UTILIZATION_THRESHOLD = 0.8  # phi_u (过载阈值)

# --- 新的平滑奖励系数 (New smooth reward coefficients) ---
ALPHA_UTIL_BONUS = 5.0      # (奖励) 鼓励高平均利用率
BETA_IMBALANCE_PENALTY = 0.5 # (惩罚) 惩罚负载不均衡 (高标准差)
GAMMA_OVERLOAD_PENALTY = 1.0 # (惩罚) 严厉惩罚 *每台* 过载的服务器
# --- 失败动作的奖励 (Rewards for failed actions) ---
R_BUSY_PENALTY = -0.08     # (保持) 针对“选了台忙碌服务器”的轻微惩罚
R_JOB_FAIL = -5.0         # (保持) 调度失败/放弃的奖励 (必须最差)

# R = (Alpha * 平均利用率) - (Beta * 负载不均衡度) - (Gamma * 过载数量)

# 6. DRL Agent 超参数 (来自 论文表4 / agent.py)
GAMMA = 0.95  # 折扣因子 (Discount factor)
LR = 0.001  # 学习率 (Learning rate)
REPLAY_MEMORY_SIZE = 50000  # 内存容量 (Memory capacity)
BATCH_SIZE = 128  # 批量尺寸 (Batch size)
EPS_START = 0.15  # epsilon 初始值 (Epsilon initial value)
EPS_END = 0.001  # epsilon 最小值 (Epsilon minimum value)
EPS_DECAY_RATE = 0.000002  # epsilon 衰减率 (Epsilon decay rate)
TARGET_UPDATE_PERIOD = 800  # 目标网络更新周期 (Target network update period)