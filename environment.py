
import config
import it_systems
import task_manager
import numpy as np


class DataCenterEnv:
    """多数据中心内部任务部署环境"""
    def __init__(self, workload_queue):
        print("[Env] Initializing Data Center Environment...")
        # 1. 初始化 IT 系统 (Initialize IT Systems)
        self.racks = [it_systems.Rack(id=i) for i in range(config.NUM_RACKS)]
        self.servers = [server for rack in self.racks for server in rack.servers]
        if len(self.servers) != config.TOTAL_SERVERS:
            raise Exception("Server count mismatch!")

        # 2. 初始化 FCFS 任务队列 (Initialize FCFS Task Queue)
        self.task_queue = workload_queue

        print(f"[Env] {config.TOTAL_SERVERS} servers in {config.NUM_RACKS} racks initialized.")
        print(f"[Env] {len(self.task_queue)} tasks loaded in FCFS queue.")

    def get_current_task(self):
        """ FCFS: 仅查看队列头的任务 (FCFS: Only look at the task at the front) """
        if not self.task_queue:
            return None
        return self.task_queue[0]

    def get_state(self):
        """
        获取论文 2.3 节定义的状态 S = {Ca, U, P, To}  不考虑温度
        """

        available_cpu = np.array([s.total_cpu - s.used_cpu for s in self.servers])
        utilization = np.array([s.utilization for s in self.servers])
        power = np.array([s.power for s in self.servers])

        # 返回字典，Agent需要将其扁平化为DQN输入
        return {
            "available_cpu": available_cpu,  # C_a
            "utilization": utilization,  # U
            "power": power,  # P
        }

    # --- (*** 应用最终的统计奖励逻辑 ***) ---
    def step(self, action):
        """
        环境的核心步骤函数
        action: 动作 A = {0, 1, ..., K}
        """

        # 1. 推进所有服务器的时间步
        for s in self.servers:
            s.step()  # 可能会有任务完成并释放资源

        # 2. 检查任务队列 (Check task queue)
        current_task = self.get_current_task()
        if current_task is None:
            # 任务全部完成 (All tasks are done)
            done = True
            return self.get_state(), 0.0, done, {"success": False, "message": "Queue empty"}

        # 3. 执行动作 A (Execute Action A)
        success = False
        message = ""
        reward = 0.0

        # 动作 0: "放弃"
        if action == 0:
            success = False
            message = "Action 0 (No-op)"
            reward = config.R_JOB_FAIL  # (现在是 -5.0)
            self.task_queue.popleft()

        # 动作 1...K: "尝试分配"
        elif 1 <= action <= config.TOTAL_SERVERS:
            server = self.servers[action - 1]  # 动作 1 对应索引 0
            # 3a. 尝试成功
            if server.can_assign_task(current_task):
                server.assign_task(current_task)
                success = True
                message = f"Assigned to server {server.id}"
                # (*** 新的奖励逻辑: 计算全局统计数据 ***)
                # 获取刚刚更新的 t+1 时刻的全局状态
                current_state_dict = self.get_state()
                util_vector = current_state_dict["utilization"]
                # 1. 奖励: 平均利用率
                mean_util = np.mean(util_vector)
                # 2. 惩罚: 负载不均衡 (标准差)
                std_dev_util = np.std(util_vector)
                # 3. 惩罚: 过载服务器数量
                overload_count = np.sum(util_vector > config.UTILIZATION_THRESHOLD)
                # R = (Alpha * Mean_Util) - (Beta * StdDev_Util) - (Gamma * Overload_Count)
                reward = (mean_util * config.ALPHA_UTIL_BONUS) - \
                         (std_dev_util * config.BETA_IMBALANCE_PENALTY) - \
                         (overload_count * config.GAMMA_OVERLOAD_PENALTY)
                # 移除任务
                self.task_queue.popleft()
            # 3b. 尝试失败 (服务器忙碌)
            else:
                success = False
                message = f"Server {server.id} cannot take task (Busy or Insufficient CPU)"
                reward = config.R_BUSY_PENALTY  # (保持 -0.1)
                # *不*移除任务，Agent 必须在下一步重试
        # 动作无效 (例如: action = 673)
        else:
            success = False
            message = f"Invalid action {action}"
            reward = config.R_JOB_FAIL  # (现在是 -5.0)
            self.task_queue.popleft()

        # 6. 检查是否结束 (Check if done)
        done = not self.task_queue

        # (!!!) 注意：返回的 self.get_state() 已经是 t+1 的状态
        return self.get_state(), reward, done, {"success": success, "message": message}