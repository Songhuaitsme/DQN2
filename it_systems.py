# -*- coding: utf-8 -*-
"""
实现论文中定义的服务器和机架模型
(Implements the Server and Rack models defined in the paper)
"""

import config


class Server:
    """
    单个服务器模型 (Single Server Model)
    严格遵循式(4)利用率 和 式(5)功耗
    (Strictly follows Eq. 4 (Utilization) and Eq. 5 (Power))
    """

    def __init__(self, id, rack_id):
        self.id = id
        self.rack_id = rack_id
        self.total_cpu = config.CPU_PER_SERVER
        self.used_cpu = 0

        # 任务信息 (Task Information)
        self.current_task = None
        self.task_remaining_duration = 0

        # 服务器状态 (Server State)
        self.utilization = 0.0
        self.power = config.P_IDLE

    def _update_utilization(self):
        """ Eq. 4: u_k = c_using / c_total """
        self.utilization = self.used_cpu / self.total_cpu

    def _update_power(self):
        """ Eq. 5: p_k = p_idle + (p_full - p_idle) * u_k """
        self.power = config.P_IDLE + (config.P_FULL - config.P_IDLE) * self.utilization

    def update_server_state(self):
        """ 更新服务器的利用率和功耗 (Update server utilization and power) """
        self._update_utilization()
        self._update_power()

    def can_assign_task(self, task):
        """
        检查服务器是否可以分配此任务
        (Check if server can be assigned this task)
        1. 服务器必须空闲 (Server must be idle)
        2. 服务器必须有足够的CPU (Server must have enough CPU)
        """
        is_idle = (self.task_remaining_duration == 0)
        has_capacity = (self.total_cpu - self.used_cpu) >= task.cpu_needed
        return is_idle and has_capacity

    def assign_task(self, task):
        """
        分配任务到此服务器 (Assign task to this server)
        """
        if not self.can_assign_task(task):
            return False

        self.used_cpu += task.cpu_needed
        self.current_task = task
        self.task_remaining_duration = task.duration
        self.update_server_state()
        return True

    def release_task(self):
        """ 任务完成，释放资源 (Task complete, release resources) """
        if self.current_task:
            self.used_cpu -= self.current_task.cpu_needed
            self.current_task = None
            self.task_remaining_duration = 0
            self.update_server_state()

    def step(self):
        """ 模拟一个时间步的前进 (Simulate one time step progression) """
        if self.task_remaining_duration > 0:
            self.task_remaining_duration -= 1
            if self.task_remaining_duration == 0:
                self.release_task()


class Rack:
    """
    机架模型，包含多个服务器
    (Rack model, contains multiple servers)
    """

    def __init__(self, id):
        self.id = id
        self.servers = [
            Server(id=(id * config.SERVERS_PER_RACK + i), rack_id=id)
            for i in range(config.SERVERS_PER_RACK)
        ]

    def get_total_power(self):
        """ 计算此机架的总功耗 (Calculate total power for this rack) """
        return sum(s.power for s in self.servers)