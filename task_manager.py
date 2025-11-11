# -*- coding: utf-8 -*-
"""
实现任务、任务划分、工作负载和FCFS队列
(Implements Tasks, Task Splitting, Workload, and FCFS Queue)
"""

import config
import random
from collections import deque


class Task:
    """ 单个（子）任务模型 (Single (sub)task model) """

    def __init__(self, cpu_needed, duration):
        if cpu_needed > config.CPU_PER_SERVER:
            raise ValueError(f"Task CPU {cpu_needed} exceeds server capacity {config.CPU_PER_SERVER}")
        if cpu_needed > config.MAX_CPU_PER_SUBTASK:
            raise ValueError(f"Task CPU {cpu_needed} exceeds subtask limit {config.MAX_CPU_PER_SUBTASK}")

        self.cpu_needed = cpu_needed
        self.duration = duration

    def __repr__(self):
        return f"<Task: CPU={self.cpu_needed}, Duration={self.duration}>"


def _task_splitter(raw_cpu_needed, raw_duration):
    """
    论文 3.2.2 节的任务划分逻辑
    (Task splitting logic from Section 3.2.2)
    """
    sub_tasks = []

    # 1. 任务是否需要划分? (Does task need splitting?)
    if raw_cpu_needed > config.CPU_PER_SERVER:
        # 论文 3.2.2: "任务需求的CPU核数超过24, 则需要划分子任务"
        # (Paper 3.2.2: "If task CPU > 24, split")

        cpu_remaining = raw_cpu_needed
        while cpu_remaining > 0:
            # 论文 3.2.2: "每个子任务需求CPU核数最大不超过8"
            # (Paper 3.2.2: "Sub-task max CPU <= 8")
            sub_cpu = min(cpu_remaining, config.MAX_CPU_PER_SUBTASK)
            sub_tasks.append(Task(sub_cpu, raw_duration))
            cpu_remaining -= sub_cpu
    else:
        # 任务不需要划分 (No split needed)
        sub_tasks.append(Task(raw_cpu_needed, raw_duration))

    return sub_tasks


def load_synthetic_workload(num_raw_tasks):
    """
    生成模拟的 LLNL Thunder 工作负载
    (Generate synthetic workload, mocking LLNL Thunder)
    """
    print(f"[Workload] Generating {num_raw_tasks} synthetic raw tasks (mock for LLNL Thunder)")

    raw_tasks = []
    for _ in range(num_raw_tasks):
        # 随机生成原始任务，允许CPU > 24 以测试划分逻辑
        # (Generate raw tasks, allowing CPU > 24 to test splitting)
        cpu = random.randint(1, 30)
        duration = random.randint(10, 50)  # 模拟的时间步 (Simulated time steps)
        raw_tasks.append({'cpu': cpu, 'duration': duration})

    # 遵循FCFS的最终任务队列 (Final FCFS task queue)
    final_task_queue = deque()

    # 严格按论文逻辑处理任务 (Process tasks strictly by paper logic)
    for raw_task in raw_tasks:
        sub_tasks = _task_splitter(raw_task['cpu'], raw_task['duration'])
        for sub_task in sub_tasks:
            final_task_queue.append(sub_task)

    print(f"[Workload] {num_raw_tasks} raw tasks split into {len(final_task_queue)} sub-tasks (FCFS).")
    return final_task_queue