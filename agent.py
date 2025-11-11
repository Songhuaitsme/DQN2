# -*- coding: utf-8 -*-
"""
Double DQN 智能体 (Agent) - TensorFlow 2.8 版本
(Double DQN Agent - TensorFlow 2.8 Version)

使用 tf.keras 和 tf.GradientTape 实现
(Implemented using tf.keras and tf.GradientTape)
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from collections import deque, namedtuple

# --- 从 config.py 导入的常量 ---
# (Constants from config.py)
NUM_RACKS = 16
SERVERS_PER_RACK = 42
TOTAL_SERVERS = NUM_RACKS * SERVERS_PER_RACK

# --- 论文表4中的超参数 ---
# (Hyperparameters from Paper's Table 4)
GAMMA = 0.99  # 折扣因子 (Discount factor)
LR = 0.001  # 学习率 (Learning rate)
REPLAY_MEMORY_SIZE = 5000  # 内存容量 (Memory capacity)
BATCH_SIZE = 128  # 批量尺寸 (Batch size)
EPS_START = 0.2  # epsilon 初始值 (Epsilon initial value)
EPS_END = 0.001  # epsilon 最小值 (Epsilon minimum value)
EPS_DECAY_RATE = 0.0000199  # epsilon 衰减率 (Epsilon decay rate)
TARGET_UPDATE_PERIOD = 30  # 目标网络更新周期 (Target network update period)

# --- 智能体维度 ---
# (Agent Dimensions)
# 状态 S = {Ca, U, P, To}
# K+K+K+N = 672+672+672+16 = 2032
# STATE_DIM = (TOTAL_SERVERS * 3) + NUM_RACKS
STATE_DIM = (TOTAL_SERVERS * 3)
# 动作 A = {0, 1, ..., K}
# K+1 = 673
ACTION_DIM = TOTAL_SERVERS + 1

# --- 经验回放池 (无需更改) ---
# (Experience Replay Buffer (No change needed))
Experience = namedtuple('Experience',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """保存一个经验 (Save an experience)"""
        # 存储的是 NumPy 数组
        # (Stores NumPy arrays)
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        """随机采样一批经验 (Randomly sample a batch of experiences)"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# --- DQN 神经网络 (tf.keras) ---
# (DQN Neural Network (tf.keras))
class DQN_Network(Model):
    """
    遵循论文 3.2.3 节的描述: 3个隐藏层
    (Follows Section 3.2.3: 3 hidden layers)
    """

    def __init__(self, output_dim):
        super(DQN_Network, self).__init__()
        hidden_1 = 512
        hidden_2 = 256
        hidden_3 = 128

        # 使用 tf.keras.layers 定义层
        # (Define layers using tf.keras.layers)
        self.fc1 = Dense(hidden_1, activation='relu')
        self.fc2 = Dense(hidden_2, activation='relu')
        self.fc3 = Dense(hidden_3, activation='relu')
        self.fc4 = Dense(output_dim, activation='linear')  # Q 值是线性的 (Q-values are linear)

    def call(self, x):
        """ 定义前向传播 (Define the forward pass) """
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.fc4(x)


# --- Double DQN 智能体 (TensorFlow) ---
# (Double DQN Agent (TensorFlow))
class DoubleDQN_Agent:
    def __init__(self):
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM

        # 初始化两个网络 (Initialize two networks)
        self.q_network = DQN_Network(self.action_dim)
        self.target_network = DQN_Network(self.action_dim)

        # 初始化优化器
        # (Initialize optimizer)
        self.optimizer = Adam(learning_rate=LR)

        # 初始化经验回放池
        # (Initialize replay buffer)
        self.memory = ReplayBuffer(REPLAY_MEMORY_SIZE)

        self.steps_done = 0
        self.epsilon = EPS_START

        # 立即构建网络，以便我们可以复制权重
        # (Build networks immediately so we can copy weights)
        self.q_network.build(input_shape=(None, self.state_dim))
        self.target_network.build(input_shape=(None, self.state_dim))
        self.update_target_network()

    def update_target_network(self):
        """
        将 Q 网络的权重复制到目标网络
        (Copy weights from Q-network to Target-network)
        """
        self.target_network.set_weights(self.q_network.get_weights())

    def _flatten_state(self, state_dict):
        """
        将环境返回的字典状态扁平化为DQN输入向量
        (Flatten the state dict from env into a vector for DQN input)
        """
        s1 = state_dict['available_cpu']
        s2 = state_dict['utilization']
        s3 = state_dict['power']
        # s4 = state_dict['rack_temperatures']

        # 将所有状态连接成一个长向量 (NumPy)
        # (Concatenate all states into one long vector (NumPy))
        return np.concatenate([s1, s2, s3], axis=0)

    def select_action(self, state_dict):
        """
        使用 Epsilon-Greedy 策略选择动作
        (Select an action using Epsilon-Greedy policy)

        state_dict: 来自环境的原始字典状态
        (state_dict: The raw dictionary state from the environment)
        """

        # 1. 计算当前的 Epsilon
        # (Calculate current Epsilon)
        self.epsilon = max(EPS_END, self.epsilon - EPS_DECAY_RATE * self.steps_done)
        self.steps_done += 1

        if random.random() > self.epsilon:
            # 2. 利用 (Exploitation)
            # 扁平化状态
            # (Flatten state)
            state_vec = self._flatten_state(state_dict)
            # 添加 batch 维度 (1, state_dim)
            # (Add batch dimension (1, state_dim))
            state_tensor = tf.convert_to_tensor(state_vec.reshape(1, -1), dtype=tf.float32)

            # 从 Q 网络获取 Q 值
            # (Get Q-values from Q-network)
            q_values = self.q_network(state_tensor)

            # 选择 Q 值最大的动作
            # (Select action with max Q-value)
            action = tf.argmax(q_values[0]).numpy()
            return action
        else:
            # 3. 探索 (Exploration)
            # 随机选择一个动作
            # (Select a random action)
            return random.randrange(self.action_dim)

    def store_experience(self, state_dict, action, reward, next_state_dict, done):
        """
        扁平化状态并存入经验回放池
        (Flatten states and store in replay buffer)
        """
        state_vec = self._flatten_state(state_dict)
        next_state_vec = self._flatten_state(next_state_dict)

        # 存储 NumPy 数组，而不是Tensors
        # (Store NumPy arrays, not Tensors)
        self.memory.push(state_vec, action, reward, next_state_vec, done)

    def learn(self):
        """
        Double DQN 的核心训练逻辑 (TensorFlow)
        (Core training logic for Double DQN (TensorFlow))
        """

        # 1. 检查是否有足够的经验
        # (Check if there is enough experience)
        if len(self.memory) < BATCH_SIZE:
            return

        # 2. 从回放池中采样
        # (Sample from replay buffer)
        experiences = self.memory.sample(BATCH_SIZE)
        batch = Experience(*zip(*experiences))

        # 3. 将 NumPy 批次转换为 Tensors
        # (Convert NumPy batches to Tensors)
        states = tf.convert_to_tensor(np.array(batch.state), dtype=tf.float32)
        actions = tf.convert_to_tensor(np.array(batch.action), dtype=tf.int64)
        rewards = tf.convert_to_tensor(np.array(batch.reward), dtype=tf.float32)
        next_states = tf.convert_to_tensor(np.array(batch.next_state), dtype=tf.float32)
        dones = tf.convert_to_tensor(np.array(batch.done), dtype=tf.float32)

        # 4. Double DQN 核心逻辑
        # (Double DQN Core Logic)

        # 4a. 使用 Q-Network 选择下一状态的最佳动作
        # (Use Q-Network to select best action for next state)
        next_actions = tf.argmax(self.q_network(next_states), axis=1, output_type=tf.int64)

        # 4b. 使用 Target-Network 评估该动作的 Q 值
        # (Use Target-Network to evaluate that action's Q-value)
        # 我们需要使用 tf.gather_nd 来选择 Q 值
        # (We need to use tf.gather_nd to select the Q-values)
        next_actions_indices = tf.stack([tf.range(BATCH_SIZE, dtype=tf.int64), next_actions], axis=1)
        next_q_values = self.target_network(next_states)
        next_q_values_selected = tf.gather_nd(next_q_values, next_actions_indices)

        # 5. 计算目标 Q 值 (y_j)
        # (Calculate target Q-value (y_j))
        expected_q_values = rewards + (GAMMA * next_q_values_selected * (1 - dones))

        # 6. 使用 tf.GradientTape 计算损失和梯度
        # (Use tf.GradientTape to calculate loss and gradients)
        with tf.GradientTape() as tape:
            # 获取所有动作的 Q 值
            # (Get Q-values for all actions)
            current_q_values_all = self.q_network(states)

            # 选择我们实际采取的动作的 Q 值
            # (Select the Q-values for the actions we actually took)
            actions_indices = tf.stack([tf.range(BATCH_SIZE, dtype=tf.int64), actions], axis=1)
            current_q_values = tf.gather_nd(current_q_values_all, actions_indices)

            # 计算损失 (MSE)
            # (Calculate loss (MSE))
            loss = tf.keras.losses.MSE(expected_q_values, current_q_values)

        # 7. 应用梯度
        # (Apply gradients)
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        # 8. 更新目标网络
        # (Update Target Network)
        # 遵循论文表4: 目标网络更新周期 = 30
        # (Follows Table 4: Target network update period = 30)
        if self.steps_done % TARGET_UPDATE_PERIOD == 0:
            self.update_target_network()