import environment
import task_manager
import config
import numpy as np
from agent import DoubleDQN_Agent
from baseline_agents import RRAgent, GreedyAgent
import os
import random
import datetime
import tensorflow as tf

# 确保可复现性
np.random.seed(42)
random.seed(42)

def run_simulation(agent, agent_name, workload, is_training=False, log_writer=None, global_step_start=0):

    # 1. 初始化环境 (使用传入的工作负载)
    env = environment.DataCenterEnv(workload.copy())  # 复制以防修改

    # 2. 如果是评估模式，DQN应使用贪心策略 (Epsilon = min)
    if isinstance(agent, DoubleDQN_Agent) and not is_training:
        agent.epsilon = config.EPS_END

    # 3. 运行仿真
    state = env.get_state()
    done = False

    # (修正) local_step 跟踪决策步数
    local_step = 0
    # (修正) total_tasks_in_loop 是我们的分母
    total_tasks_in_loop = len(workload)

    # 收集指标
    step_metrics = {
        'rewards': [],
        'util_mean': [],
        'overload_counts': []
    }

    # --- (*** 关键修改 1: 添加 'tasks_processed' 跟踪 ***) ---
    tasks_scheduled = 0
    # 我们需要跟踪从队列中移除的任务总数
    tasks_processed = 0
    # --- (*** 修改结束 ***) ---

    while not done:
        current_task = env.get_current_task()
        if current_task is None:
            break

        # 1. Agent 选择动作
        if agent_name == "DoubleDQN":
            action = agent.select_action(state,current_task)
        else:
            action = agent.select_action(state, current_task)

        # 2. 环境执行动作
        next_state, reward, done, info = env.step(action)
        loss_value = None  # 初始化 loss

        # (*** 关键修改: 在 store 之前获取 "下一个" 任务 ***)
        next_task = env.get_current_task()  # (Task T+1)
        # (如果 done=True, next_task 将为 None, _flatten_state 会处理)

        # 3. 如果是训练模式，DQN需要存储和学习
        if agent_name == "DoubleDQN" and is_training:
            agent.store_experience(state, action, reward, next_state, done, current_task, next_task)
            loss_value = agent.learn()

        # 4. 更新状态 (这在上次修改中已修正)
        state = next_state

        # 5. 收集 t+1 时刻的指标
        step_metrics['rewards'].append(reward)
        current_util_mean = np.mean(state["utilization"])
        current_overloads = np.sum(state["utilization"] > config.UTILIZATION_THRESHOLD)
        step_metrics['util_mean'].append(current_util_mean)
        step_metrics['overload_counts'].append(current_overloads)

        # --- (*** 关键修改 2: 更新 'tasks_processed' ***) ---
        if info.get("success"):
            tasks_scheduled += 1
            tasks_processed += 1  # 成功是一个已处理的任务
        elif reward == config.R_JOB_FAIL:
            # R_JOB_FAIL (放弃/无效) 也会移除任务
            tasks_processed += 1
            # (注意: R_BUSY_PENALTY 不会移除任务，所以 'tasks_processed' 不增加)
        # --- (*** 修改结束 ***) ---

        # 6. 使用“全局步数”来记录 TensorBoard
        current_global_step = global_step_start + local_step

        if log_writer is not None:
            # (日志记录逻辑已在上次修复)
            with log_writer.as_default():
                tf.summary.scalar('Environment/Reward', reward, step=current_global_step)
                tf.summary.scalar('Environment/Avg_Utilization_Percent', current_util_mean * 100.0,
                                  step=current_global_step)
                tf.summary.scalar('Environment/Overload_Servers', current_overloads,
                                  step=current_global_step)

                if agent_name == "DoubleDQN" and is_training:
                    tf.summary.scalar('Agent/Epsilon', agent.epsilon, step=current_global_step)
                    if loss_value is not None:
                        tf.summary.scalar('Agent/Loss', loss_value, step=current_global_step)

        # 7. 增加决策步数
        local_step += 1

        # 8.--- (*** 关键修改 3: 修正打印语句 ***) ---
        if is_training:
            # (修正) 进度条现在使用 'tasks_processed'
            progress = (tasks_processed / total_tasks_in_loop) * 100
            loss_str = f"{loss_value:.4f}" if loss_value is not None else "N/A"
            print(f"  [训练中]... 进度: {tasks_processed:>5}/{total_tasks_in_loop} ({progress:>5.1f}%) | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Loss: {loss_str:<8} | "
                  f"Reward: {reward:>6.2f}", end="\r")
        # --- (*** 修改结束 ***) ---

    if is_training:
        print()  # 结束 \r 状态，移动到下一行

    # ... (函数其余部分无需更改) ...

    # 9. 计算最终平均指标
    total_steps = len(step_metrics['rewards'])
    if total_steps == 0:
        results = {"Agent": agent_name, "Mode": "Training" if is_training else "Evaluation", "Avg Reward": 0,
                   "Avg Util (%)": 0, "Avg Overloads": 0, "Tasks Scheduled": 0}
    else:
        results = {
            "Agent": agent_name,
            "Mode": "Training" if is_training else "Evaluation",
            "Avg Reward": np.mean(step_metrics['rewards']),
            "Avg Util (%)": np.mean(step_metrics['util_mean']) * 100.0,
            "Avg Overloads": np.mean(step_metrics['overload_counts']),
            "Tasks Scheduled": tasks_scheduled
        }

    # 10. 返回 results 和 *更新后*的全局步数
    return results, (global_step_start + local_step)

# --- 运行主程序 ---
if __name__ == "__main__":

    # --- 1. 加载工作负载 ---
    print("\n--- 阶段 0: 正在加载/生成工作负载 ---")
    train_workload_tasks = task_manager.get_train_workload()
    eval_workload_tasks = task_manager.get_eval_workload()

    # 2. 初始化所有 Agents
    dqn_agent = DoubleDQN_Agent()
    rr_agent = RRAgent(config.TOTAL_SERVERS)
    greedy_agent = GreedyAgent(config.TOTAL_SERVERS)

    # --- 阶段 0b: 设置 TensorBoard ---
    if not os.path.exists('logs'):
        os.makedirs('logs')

    log_dir = "logs/dqn-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_writer = tf.summary.create_file_writer(log_dir)
    print(f"\n--- TensorBoard 日志将保存到: {log_dir} ---")
    print(f"--- (在终端中运行 'tensorboard --logdir logs' 来查看) ---")

    # --- 阶段 1: 训练 DQN Agent ---
    print("\n--- 阶段 1: 正在训练 DoubleDQN Agent ---")

    NUM_EPISODES = 250  # 运行 200 轮
    global_step = 0  # 初始化全局步数

    for episode in range(NUM_EPISODES):
        print(f"\n--- 开始情节 {episode + 1} / {NUM_EPISODES} (当前总步数: {global_step}) ---")

        train_results, global_step = run_simulation(dqn_agent, "DoubleDQN", train_workload_tasks.copy(),
                                                    is_training=True, log_writer=train_writer,
                                                    global_step_start=global_step)

        # 记录“每个情节”的平均数据
        with train_writer.as_default():
            tf.summary.scalar('Episode/Avg_Reward', train_results['Avg Reward'], step=episode)
            tf.summary.scalar('Episode/Avg_Overloads', train_results['Avg Overloads'], step=episode)
            tf.summary.scalar('Episode/Avg_Utilization_Percent', train_results['Avg Util (%)'], step=episode)

    print("\n--- DQN 训练完成。---")

    # --- 阶段 1b: 保存模型 ---
    if not os.path.exists('models'):
        os.makedirs('models')

    model_save_path = "models/dqn_trained_weights.h5"
    dqn_agent.save_model_weights(model_save_path)

    # --- 阶段 2: 评估所有 Agents ---
    print(f"\n--- 阶段 2: 正在评估三种算法 (使用 500 任务) ---")

    print("  正在评估: DoubleDQN (Trained, Epsilon=Min)...")
    dqn_eval_results, _ = run_simulation(dqn_agent, "DoubleDQN", eval_workload_tasks.copy(), is_training=False)

    print("  正在评估: Round Robin (RR)...")
    rr_eval_results, _ = run_simulation(rr_agent, "RR", eval_workload_tasks.copy(), is_training=False)

    print("  正在评估: Greedy...")
    greedy_eval_results, _ = run_simulation(greedy_agent, "Greedy", eval_workload_tasks.copy(), is_training=False)

    # --- 阶段 3: 打印对比结果 (模拟表5) ---
    print("\n--- 评估完成：对比结果 (模拟 论文表5) ---")
    print(
        f"{'Agent':<12} | {'Avg Reward':<12} | {'Avg Overloads':<15} | {'Avg Util (%)':<15} | {'Tasks Scheduled':<15}")
    print("-" * 75)

    for res in [dqn_eval_results, rr_eval_results, greedy_eval_results]:
        print(f"{res['Agent']:<12} | {res['Avg Reward']:<12.2f} | "
              f"{res['Avg Overloads']:<15.2f} | "
              f"{res['Avg Util (%)']:<15.2f} | {res['Tasks Scheduled']:<15}")

    print("\n注意: 'Avg Overloads' 指在任意时间步，利用率超过 80% 的服务器的平均数量。")