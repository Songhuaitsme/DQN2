# -*- coding: utf-8 -*-
"""
主比较程序 (Main Comparison Program)

1. 训练 DoubleDQN Agent (使用 3000 任务)
2. 评估已训练的 DoubleDQN、RR 和 Greedy (使用 500 任务)
3. 打印对比结果 (模拟表5)
"""

import environment
import task_manager
import config
import numpy as np
from agent import DoubleDQN_Agent
from baseline_agents import RRAgent, GreedyAgent
import os
import random

# 确保可复现性
np.random.seed(42)
random.seed(42)


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # 确保使用 CPU

def run_simulation(agent, agent_name, workload, is_training=False):
    """
    运行一次完整的仿真（训练或评估）
    """

    # 1. 初始化环境 (使用传入的工作负载)
    env = environment.DataCenterEnv(workload.copy())  # 复制以防修改

    # 2. 如果是评估模式，DQN应使用贪心策略 (Epsilon = min)
    if isinstance(agent, DoubleDQN_Agent) and not is_training:
        agent.epsilon = config.EPS_END

    # 3. 运行仿真
    state = env.get_state()
    done = False

    # 收集指标
    step_metrics = {
        'rewards': [],
        'rack_temps_mean': [],
        'util_mean': [],
        'overload_counts': []
    }
    tasks_scheduled = 0

    while not done:
        current_task = env.get_current_task()
        if current_task is None:
            break

        # 1. Agent 选择动作
        # DQN Agent 仅需要 state
        # Baseline Agents 需要 state 和 task
        if agent_name == "DoubleDQN":
            action = agent.select_action(state)
        else:
            action = agent.select_action(state, current_task)

        # 2. 环境执行动作
        next_state, reward, done, info = env.step(action)

        # 3. 如果是训练模式，DQN需要存储和学习
        if agent_name == "DoubleDQN" and is_training:
            agent.store_experience(state, action, reward, next_state, done)
            agent.learn()

        # 4. 收集指标
        step_metrics['rewards'].append(reward)
        # step_metrics['rack_temps_mean'].append(np.mean(state["rack_temperatures"]))
        step_metrics['util_mean'].append(np.mean(state["utilization"]))
        # 计算过载的服务器数量  (阈值来自 config [cite: 299])
        overloads = np.sum(state["utilization"] > config.UTILIZATION_THRESHOLD)
        step_metrics['overload_counts'].append(overloads)

        if info.get("success"):
            tasks_scheduled += 1

        # 更新状态
        state = next_state

    # 5. 计算最终平均指标
    total_steps = len(step_metrics['rewards'])
    if total_steps == 0:
        return {
            "Agent": agent_name,
            "Mode": "Training" if is_training else "Evaluation",
            "Avg Reward": 0,
            "Avg Temp (°C)": 0,
            "Avg Util (%)": 0,
            "Avg Overloads": 0,
            "Tasks Scheduled": 0
        }

    results = {
        "Agent": agent_name,
        "Mode": "Training" if is_training else "Evaluation",
        "Avg Reward": np.mean(step_metrics['rewards']),
        "Avg Temp (°C)": np.mean(step_metrics['rack_temps_mean']),
        "Avg Util (%)": np.mean(step_metrics['util_mean']) * 100.0,
        # 我们计算的是平均每一步的过载服务器数量
        "Avg Overloads": np.mean(step_metrics['overload_counts']),
        "Tasks Scheduled": tasks_scheduled
    }
    return results


# --- 运行主程序 ---
if __name__ == "__main__":

    # 1. 加载工作负载
    # 论文: 3000个任务用于训练
    train_workload = task_manager.load_synthetic_workload(3000)
    # 论文: 500个任务用于验证
    eval_workload = task_manager.load_synthetic_workload(500)

    # 2. 初始化所有 Agents
    dqn_agent = DoubleDQN_Agent()
    rr_agent = RRAgent(config.TOTAL_SERVERS)
    greedy_agent = GreedyAgent(config.TOTAL_SERVERS)

    # --- 阶段 1: 训练 DQN Agent ---
    print("\n--- 阶段 1: 正在训练 DoubleDQN Agent (3000 任务) ---")
    # 传入 train_workload 并设置 is_training=True
    train_results = run_simulation(dqn_agent, "DoubleDQN", train_workload, is_training=True)
    print("DQN 训练完成。")
    print(f"  训练结果: Avg Reward: {train_results['Avg Reward']:.2f}, "
          f"Tasks Scheduled: {train_results['Tasks Scheduled']}/3000")

    # --- 阶段 2: 评估所有 Agents ---
    print(f"\n--- 阶段 2: 正在评估三种算法 (使用 500 任务) ---")

    # 评估 1: 已训练的 DoubleDQN
    print("  正在评估: DoubleDQN (Trained, Epsilon=Min)...")
    dqn_eval_results = run_simulation(dqn_agent, "DoubleDQN", eval_workload, is_training=False)

    # 评估 2: RR Agent
    print("  正在评估: Round Robin (RR)...")
    rr_eval_results = run_simulation(rr_agent, "RR", eval_workload, is_training=False)

    # 评估 3: Greedy Agent
    print("  正在评估: Greedy...")
    greedy_eval_results = run_simulation(greedy_agent, "Greedy", eval_workload, is_training=False)

    # --- 阶段 3: 打印对比结果 (模拟表5) ---
    print("\n--- 评估完成：对比结果 (模拟 论文表5) ---")

    print(
        f"{'Agent':<12} | {'Avg Reward':<12} | {'Avg Temp (°C)':<15} | {'Avg Overloads':<15} | {'Avg Util (%)':<15} | {'Tasks Scheduled':<15}")
    print("-" * 90)

    for res in [dqn_eval_results, rr_eval_results, greedy_eval_results]:
        print(f"{res['Agent']:<12} | {res['Avg Reward']:<12.2f} | "
              f"{res['Avg Temp (°C)']:<15.2f} | {res['Avg Overloads']:<15.2f} | "
              f"{res['Avg Util (%)']:<15.2f} | {res['Tasks Scheduled']:<15}")

    print("\n注意: 'Avg Overloads' 指在任意时间步，利用率超过 80% [cite: 299] 的服务器的平均数量。")