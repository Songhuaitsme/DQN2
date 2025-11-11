# -*- coding: utf-8 -*-
"""
DRL 智能体的交互环境 (Gym-like environment for DRL Agent)
严格遵循 2.3 节的 S, A, R 定义
(Strictly follows S, A, R definitions from Section 2.3)
"""

import config
import it_systems
import task_manager
import thermal_mock
import numpy as np


class DataCenterEnv:
    """
    多数据中心内部任务部署环境
    (Data Center Internal Task Deployment Environment)
    """

    def __init__(self, workload_queue):
        print("[Env] Initializing Data Center Environment...")
        # 1. 初始化 IT 系统 (Initialize IT Systems)
        self.racks = [it_systems.Rack(id=i) for i in range(config.NUM_RACKS)]
        # 扁平化的服务器列表，用于 Agent 索引 (Flattened server list for Agent)
        self.servers = [server for rack in self.racks for server in rack.servers]
        if len(self.servers) != config.TOTAL_SERVERS:
            raise Exception("Server count mismatch!")

        # 2. 初始化 FCFS 任务队列 (Initialize FCFS Task Queue)
        self.task_queue = workload_queue

        # 3. 初始化 CFD 热力模拟 (Initialize CFD Thermal Mock)
        # self.cfd = thermal_mock.SimpleCFD()

        print(f"[Env] {config.TOTAL_SERVERS} servers in {config.NUM_RACKS} racks initialized.")
        print(f"[Env] {len(self.task_queue)} tasks loaded in FCFS queue.")

    def get_current_task(self):
        """ FCFS: 仅查看队列头的任务 (FCFS: Only look at the task at the front) """
        if not self.task_queue:
            return None
        return self.task_queue[0]

    def get_state(self):
        """
        获取论文 2.3 节定义的状态 S = {Ca, U, P, To}
        (Get state S = {Ca, U, P, To} as defined in Section 2.3)
        """
        # S_IT: K-dimensional vectors
        available_cpu = np.array([s.total_cpu - s.used_cpu for s in self.servers])
        utilization = np.array([s.utilization for s in self.servers])
        power = np.array([s.power for s in self.servers])

        # S_th: N-dimensional vector
        # temperatures = self.cfd.get_temperatures()

        # 返回字典，Agent需要将其扁平化为DQN输入
        # (Return dict, Agent must flatten this for DQN input)
        return {
            "available_cpu": available_cpu,  # C_a
            "utilization": utilization,  # U
            "power": power,  # P
            # "rack_temperatures": temperatures  # T_o
        }

    def _calculate_reward(self, success):
        """
        计算论文 2.3 节定义的奖励 R (Eq. 13 & 7)
        (Calculate Reward R as defined in Section 2.3 (Eq. 13 & 7))
        """

        # Eq. 13: 调度失败 (Scheduling failed)
        if not success:
            return config.R_JOB_FAIL

            # Eq. 7: 调度成功，计算 eta (Scheduling success, calculate eta)
        eta = config.R_JOB_SUCCESS

        # # 温度惩罚 (alpha) (Temperature Penalty)
        # temps = self.cfd.get_temperatures()
        # temp_overshoot = np.maximum(0, temps - config.TEMPERATURE_THRESHOLD)
        # temp_penalty = np.mean(temp_overshoot)
        # eta -= config.ALPHA_TEMP_PENALTY * temp_penalty

        # 负载惩罚 (beta) (Load Penalty)
        utils = np.array([s.utilization for s in self.servers])
        load_overshoot = np.maximum(0, utils - config.UTILIZATION_THRESHOLD)
        load_penalty = np.mean(load_overshoot)
        eta -= config.BETA_LOAD_PENALTY * load_penalty

        return eta

    def step(self, action):
        """
        环境的核心步骤函数 (Core step function for the environment)
        action: 动作 A = {0, 1, ..., K}
        """

        # 1. 推进所有服务器的时间步 (Advance time step for all servers)
        for s in self.servers:
            s.step()  # 可能会有任务完成并释放资源 (Tasks may complete)

        # 2. 检查任务队列 (Check task queue)
        current_task = self.get_current_task()
        if current_task is None:
            # 任务全部完成 (All tasks are done)
            done = True
            return self.get_state(), 0.0, done, {"success": False, "message": "Queue empty"}

        # 3. 执行动作 A (Execute Action A)
        success = False
        message = ""

        # 动作 0: 不分配 (Action 0: No-op / Do not assign)
        if action == 0:
            success = False
            message = "Action 0 (No-op)"

        # 动作 1...K: 分配到服务器 (Action 1...K: Assign to server)
        elif 1 <= action <= config.TOTAL_SERVERS:
            server = self.servers[action - 1]  # 动作 1 对应索引 0 (Action 1 maps to index 0)
            if server.can_assign_task(current_task):
                server.assign_task(current_task)
                self.task_queue.popleft()  # FCFS: 任务成功，移出队列
                success = True
                message = f"Assigned to server {server.id}"
            else:
                success = False
                message = f"Server {server.id} cannot take task (Busy or Insufficient CPU)"
        else:
            # 无效动作 (Invalid action)
            success = False
            message = f"Invalid action {action}"

        # # 4. 更新热力环境 (Update thermal environment)
        # rack_powers = np.array([rack.get_total_power() for rack in self.racks])
        # self.cfd.update_rack_temps(rack_powers)

        # 5. 计算奖励 R (Calculate Reward R)
        reward = self._calculate_reward(success)

        # 6. 检查是否结束 (Check if done)
        done = not self.task_queue

        return self.get_state(), reward, done, {"success": success, "message": message}