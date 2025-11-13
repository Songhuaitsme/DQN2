import config
import random
from collections import deque
import json


class Task:
    """ 单个（子）任务模型 (Single (sub)task model) """

    def __init__(self, cpu_needed, duration):
        if cpu_needed > config.CPU_PER_SERVER:
            raise ValueError(f"Task CPU {cpu_needed} exceeds server capacity {config.CPU_PER_SERVER}")

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
        cpu_remaining = raw_cpu_needed
        while cpu_remaining > 0:
            # 论文 3.2.2: "每个子任务需求CPU核数最大不超过8"
            sub_cpu = min(cpu_remaining, config.MAX_CPU_PER_SUBTASK)
            sub_tasks.append(Task(sub_cpu, raw_duration))
            cpu_remaining -= sub_cpu
    else:
        # 任务不需要划分 (No split needed)
        sub_tasks.append(Task(raw_cpu_needed, raw_duration))

    return sub_tasks


def _generate_and_save_workload(num_raw_tasks,filename):
    """
    生成模拟的 LLNL Thunder 工作负载
    """
    print(f"[Workload] 正在为 {filename} 生成 {num_raw_tasks} 个合成原始任务...")

    raw_tasks = []
    for _ in range(num_raw_tasks):
        # 随机生成原始任务，允许CPU > 24 以测试划分逻辑
        cpu = random.randint(1, 30)
        duration = random.randint(10, 50)  # 模拟的时间步 (Simulated time steps)
        raw_tasks.append({'cpu': cpu, 'duration': duration})

    # --- 保存逻辑现在在这里 ---
    print(f"[Workload] 正在保存原始任务到 {filename}...")
    try:
        with open(filename, 'w') as f:
            json.dump(raw_tasks, f, indent=2)
        print(f"[Workload] 已成功保存 {filename}。")
    except Exception as e:
        print(f"[Workload] 保存文件时出错: {e}")
    # --- 保存结束 ---

    # 遵循FCFS的最终任务队列 (Final FCFS task queue)
    final_task_queue = deque()

    # 严格按论文逻辑处理任务
    for raw_task in raw_tasks:
        sub_tasks = _task_splitter(raw_task['cpu'], raw_task['duration'])
        for sub_task in sub_tasks:
            final_task_queue.append(sub_task)

    print(f"[Workload] {num_raw_tasks} 个原始任务被拆分为 {len(final_task_queue)} 个子任务 (FCFS)。")
    return final_task_queue

# --- 3. 这是 main.py 将调用的两个新函数 ---

def get_train_workload():
    """
    获取训练工作负载 (3000个任务) 并保存到文件。
    """
    print("\n--- 正在准备训练工作负载 ---")
    return _generate_and_save_workload(
        num_raw_tasks=600,
        filename="train_workload_raw.json"
    )

def get_eval_workload():
    """
    获取评估工作负载 (500个任务) 并保存到文件。
    """
    print("\n--- 正在准备评估工作负载 ---")
    return _generate_and_save_workload(
        num_raw_tasks=150,
        filename="eval_workload_raw.json"
    )