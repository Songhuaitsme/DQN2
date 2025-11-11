# -*- coding: utf-8 -*-
"""
存放论文中定义的所有常量和超参数
(Config: Store all constants and hyperparameters from the paper)
"""

# 1. 硬件配置 (来自 3.2.2 节)
# (Hardware Config from Section 3.2.2)
NUM_RACKS = 16
SERVERS_PER_RACK = 42
TOTAL_SERVERS = NUM_RACKS * SERVERS_PER_RACK
CPU_PER_SERVER = 24  # c_k^total

# 2. 功耗模型参数 (来自 3.2.2 节)
# (Power Model Params from Section 3.2.2)
P_IDLE = 100.0  # W (p_idle)
P_FULL = 300.0  # W (p_full)

# 3. 任务约束 (来自 1.2.1 和 3.2.2 节)
# (Task Constraints from Section 1.2.1 & 3.2.2)
# 任务CPU > 24 时需划分 (Task split if CPU > 24)
# 划分后的子任务CPU核数最大不超过 8 (Sub-task max CPU <= 8)s
MAX_CPU_PER_SUBTASK = 8

# 4. CFD 模拟参数 (来自 3.2.2 节)
# (CFD Mock Params from Section 3.2.2)
# ACU_SUPPLY_TEMP = 20.0  # °C (ACU平均送风温度)

# 5. DRL 奖励函数参数 (来自 2.3 和 3.2.4 节)
# (DRL Reward Params from Section 2.3 & 3.2.4)
UTILIZATION_THRESHOLD = 0.8  # phi_u (过载阈值)
# TEMPERATURE_THRESHOLD = 27.0 # phi_T (温度阈值，论文未指定，设为27°C)

# 奖励函数超参数 (需要调优)
# (Reward hyperparameters (tuning needed))
R_JOB_SUCCESS = 10.0  # r_job (任务成功的基础奖励)
# ALPHA_TEMP_PENALTY = 1.0  # alpha (温度惩罚系数)
BETA_LOAD_PENALTY = 1.0   # beta (负载惩罚系数)
R_JOB_FAIL = -1.0         # 调度失败的奖励 (Reward for failed scheduling)