# -*- coding: utf-8 -*-
"""
实现论文 [cite: 292] 中用于对比的基线算法：
1. RR (Round Robin / 轮询) [cite: 293]
2. Greedy (贪心) [cite: 297]
"""

import numpy as np
import config


class RRAgent:
    """
    轮询调度 (Round Robin) 智能体
    论文 3.2.4 节 [cite: 293] 描述:
    "RR每次以循环方式将任务队列前面的任务分配给数据中心的服务器。"
    "如果当前服务器没有足够空闲CPU处理任务, RR将跳过该服务器并检查下一个服务器"
    """

    def __init__(self, total_servers):
        self.total_servers = total_servers
        # 记录下一个要检查的服务器索引
        self.next_server_idx = 0

    def select_action(self, state_dict, task):
        """
        选择一个动作。

        Args:
            state_dict (dict): 来自环境的 S = {Ca, U, P, To} [cite: 188]
            task (Task): 来自 task_manager 的当前任务

        Returns:
            int: 动作 A = {0, 1, ..., K}
        """
        available_cpu = state_dict['available_cpu']
        utilization = state_dict['utilization']

        # 从上一个索引开始，遍历所有服务器
        start_idx = self.next_server_idx
        for i in range(self.total_servers):
            server_idx = (start_idx + i) % self.total_servers

            # 检查服务器是否满足 `can_assign_task` 的两个条件
            # 1. 是否空闲 (我们推断 utilization == 0.0 意味着空闲)
            is_idle = (utilization[server_idx] == 0.0)
            # 2. 是否有足够容量
            has_capacity = (available_cpu[server_idx] >= task.cpu_needed)

            if is_idle and has_capacity:
                # 找到了服务器
                # 更新下一次开始的位置
                self.next_server_idx = (server_idx + 1) % self.total_servers
                # 返回动作 (索引+1)
                return server_idx + 1

        # 如果遍历一圈也没找到，更新索引并返回 0 (No-op)
        self.next_server_idx = (start_idx + 1) % self.total_servers
        return 0


class GreedyAgent:
    """
    贪心调度 (Greedy) 智能体
    论文 3.2.4 节 [cite: 297] 描述:
    "试图减少为作业服务的服务器数量。"
    "首先确定一组合格的服务器"

    注意：论文 [cite: 297] 提到“分配给当前资源利用率最高的服务器”，
    但这与环境的 `can_assign_task` 约束（服务器必须空闲）相冲突，
    因为所有空闲服务器的利用率都为 0。

    因此，我们实现一个符合“试图减少...服务器数量” [cite: 297] 精神的
    "Greedy Packer" (贪心打包器)：
    它总是试图从服务器 0 开始按顺序填充服务器。
    """

    def __init__(self, total_servers):
        self.total_servers = total_servers

    def select_action(self, state_dict, task):
        """
        选择一个动作。

        Args:
            state_dict (dict): 来自环境的 S = {Ca, U, P, To}
            task (Task): 来自 task_manager 的当前任务

        Returns:
            int: 动作 A = {0, 1, ..., K}
        """
        available_cpu = state_dict['available_cpu']
        utilization = state_dict['utilization']

        # 总是从 0 开始搜索，找到第一个可用的服务器
        for server_idx in range(self.total_servers):

            # 检查服务器是否满足 `can_assign_task` 的两个条件
            is_idle = (utilization[server_idx] == 0.0)
            has_capacity = (available_cpu[server_idx] >= task.cpu_needed)

            if is_idle and has_capacity:
                # 找到了，立即返回 (贪心)
                return server_idx + 1

        # 遍历所有服务器都无法分配
        return 0