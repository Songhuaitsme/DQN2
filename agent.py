
import tensorflow as tf
import os
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from collections import deque, namedtuple
import config

# --- 所有常量都从 config 读取 ---
NUM_RACKS = config.NUM_RACKS
SERVERS_PER_RACK = config.SERVERS_PER_RACK
TOTAL_SERVERS = config.TOTAL_SERVERS
GAMMA = config.GAMMA
LR = config.LR
REPLAY_MEMORY_SIZE = config.REPLAY_MEMORY_SIZE
BATCH_SIZE = config.BATCH_SIZE
EPS_START = config.EPS_START
EPS_END = config.EPS_END
EPS_DECAY_RATE = config.EPS_DECAY_RATE
TARGET_UPDATE_PERIOD = config.TARGET_UPDATE_PERIOD



# --- 智能体维度 ---
# (*** 关键修改：从 config 读取 STATE_DIM ***)
STATE_DIM = config.STATE_DIM
ACTION_DIM = TOTAL_SERVERS + 1

# --- 经验回放池 (无需更改) ---
Experience = namedtuple('Experience',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """保存一个经验 (Save an experience)"""
        # 存储的是 NumPy 数组
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        """随机采样一批经验 (Randomly sample a batch of experiences)"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# --- DQN 神经网络 (tf.keras) ---
class DQN_Network(Model):
    """
    遵循论文 3.2.3 节的描述: 3个隐藏层
    """

    def __init__(self, output_dim):
        super(DQN_Network, self).__init__()
        hidden_1 = 512
        hidden_2 = 256
        hidden_3 = 128

        # 使用 tf.keras.layers 定义层
        self.fc1 = Dense(hidden_1, activation='relu')
        self.fc2 = Dense(hidden_2, activation='relu')
        self.fc3 = Dense(hidden_3, activation='relu')
        self.fc4 = Dense(output_dim, activation='linear')  # Q 值是线性的 (Q-values are linear)

    def call(self, x):
        """ 定义前向传播 """
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.fc4(x)


# --- Double DQN 智能体 (TensorFlow) ---
class DoubleDQN_Agent:
    def __init__(self):
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM

        # 初始化两个网络
        self.q_network = DQN_Network(self.action_dim)
        self.target_network = DQN_Network(self.action_dim)

        # 初始化优化器
        self.optimizer = Adam(learning_rate=LR)

        # 初始化经验回放池
        self.memory = ReplayBuffer(REPLAY_MEMORY_SIZE)

        self.steps_done = 0
        self.epsilon = EPS_START

        # 立即构建网络，以便我们可以复制权重
        self.q_network.build(input_shape=(None, self.state_dim))
        self.target_network.build(input_shape=(None, self.state_dim))
        self.update_target_network()

    def save_model_weights(self, filepath):
        """保存 Q-Network 的权重"""
        print(f"\n[Agent] 正在保存模型权重到: {filepath}")
        self.q_network.save_weights(filepath)
        print("[Agent] 保存完毕。")

    def load_model_weights(self, filepath):
        """加载 Q-Network 的权重"""
        if os.path.exists(filepath):
            print(f"\n[Agent] 正在从 {filepath} 加载模型权重...")
            self.q_network.load_weights(filepath)
            # 加载权重后，立即同步目标网络
            self.update_target_network()
            print("[Agent] 加载完毕。")
        else:
            print(f"\n[Agent] 警告: 找不到模型文件 {filepath}。将使用随机初始化的网络。")

    def update_target_network(self):
        """
        将 Q 网络的权重复制到目标网络
        """
        self.target_network.set_weights(self.q_network.get_weights())

    def _flatten_state(self, state_dict,task):
        """
        将环境返回的字典状态扁平化为DQN输入向量
        """

        # --- 修改开始 ---
        # 归一化处理 (Normalization)
        # CPU: 0~26 -> 0~1
        s1 = state_dict['available_cpu'] / config.CPU_PER_SERVER
        # Utilization: 0~1 -> 0~1 (无需修改)
        s2 = state_dict['utilization']
        # Power: 100~300 -> 0~1 (除以最大功率)
        s3 = state_dict['power'] / config.P_FULL

        # 1. 扁平化服务器状态 (意大利面式)
        server_vec = np.stack([s1, s2, s3], axis=1).flatten().astype(np.float32)
        # 2. 扁平化任务状态
        if task is not None:
            # 有任务，使用它的 CPU 和持续时间
            task_cpu = task.cpu_needed / config.CPU_PER_SERVER
            task_dur = task.duration / 100.0
            task_vec = np.array([task_cpu,task_dur], dtype=np.float32)
        else:
            # 没有任务 (例如在 episode 结束时)
            task_vec = np.array([0.0, 0.0], dtype=np.float32)

        # 3. 合并
        return np.concatenate([server_vec, task_vec], axis=0)



    def select_action(self, state_dict, current_task):
        """
        使用 Epsilon-Greedy 策略选择动作
        (Select an action using Epsilon-Greedy policy)

        state_dict: 来自环境的原始字典状态
        """

        # 1. 计算当前的 Epsilon
        self.epsilon = max(EPS_END, self.epsilon - EPS_DECAY_RATE )
        self.steps_done += 1

        if random.random() > self.epsilon:
        # 2. 利用 (Exploitation)
            # 扁平化状态
            state_vec = self._flatten_state(state_dict, current_task)
            # 添加 batch 维度 (1, state_dim)
            state_tensor = tf.convert_to_tensor(state_vec.reshape(1, -1), dtype=tf.float32)
            # 从 Q 网络获取 Q 值
            q_values = self.q_network(state_tensor)
            # 选择 Q 值最大的动作
            action = tf.argmax(q_values[0]).numpy()
            return action
        else:
        # 3. 探索 (Exploration)
            # 随机选择一个动作
            return random.randrange(self.action_dim)

    def store_experience(self, state_dict, action, reward, next_state_dict, done,current_task, next_task):
        """
        扁平化状态并存入经验回放池
        """
        state_vec = self._flatten_state(state_dict,current_task)
        next_state_vec = self._flatten_state(next_state_dict,next_task)

        # 存储 NumPy 数组，而不是Tensors
        self.memory.push(state_vec, action, reward, next_state_vec, done)

    def learn(self):
        """
        Double DQN 的核心训练逻辑 (TensorFlow)
        """

        # 1. 检查是否有足够的经验
        if len(self.memory) < BATCH_SIZE:
            return None

        # 2. 从回放池中采样
        experiences = self.memory.sample(BATCH_SIZE)
        batch = Experience(*zip(*experiences))

        # 3. 将 NumPy 批次转换为 Tensors
        states = tf.convert_to_tensor(np.array(batch.state), dtype=tf.float32)
        actions = tf.convert_to_tensor(np.array(batch.action), dtype=tf.int64)
        rewards = tf.convert_to_tensor(np.array(batch.reward), dtype=tf.float32)
        next_states = tf.convert_to_tensor(np.array(batch.next_state), dtype=tf.float32)
        dones = tf.convert_to_tensor(np.array(batch.done), dtype=tf.float32)

        # 4. Double DQN 核心逻辑
        # 4a. 使用 Q-Network 选择下一状态的最佳动作
        next_actions = tf.argmax(self.q_network(next_states), axis=1, output_type=tf.int64)

        # 4b. 使用 Target-Network 评估该动作的 Q 值
        # 我们需要使用 tf.gather_nd 来选择 Q 值
        next_actions_indices = tf.stack([tf.range(BATCH_SIZE, dtype=tf.int64), next_actions], axis=1)
        next_q_values = self.target_network(next_states)
        next_q_values_selected = tf.gather_nd(next_q_values, next_actions_indices)

        # 5. 计算目标 Q 值 (y_j)
        expected_q_values = rewards + (GAMMA * next_q_values_selected * (1 - dones))

        # 6. 使用 tf.GradientTape 计算损失和梯度
        with tf.GradientTape() as tape:
            # 获取所有动作的 Q 值
            current_q_values_all = self.q_network(states)
            # 选择我们实际采取的动作的 Q 值
            actions_indices = tf.stack([tf.range(BATCH_SIZE, dtype=tf.int64), actions], axis=1)
            current_q_values = tf.gather_nd(current_q_values_all, actions_indices)
            # 计算损失 (MSE)
            loss = tf.keras.losses.MSE(expected_q_values, current_q_values)

        # 7. 应用梯度
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        # 8. 更新目标网络
        # 遵循论文表4: 目标网络更新周期 = 30
        if self.steps_done % TARGET_UPDATE_PERIOD == 0:
            self.update_target_network()

        return loss.numpy()