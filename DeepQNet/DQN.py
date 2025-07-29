import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque # 双端队列，用于回放缓冲区
import random
import matplotlib.pyplot as plt
import time

from model import QLinear, QResNet # 导入自定义的 Q 网络模型

# --- 定义回放缓冲区 (Replay Buffer) ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity) # 双端队列，达到容量时自动移除最老经验

    def push(self, state, action, reward, next_state, done):
        """添加经验到缓冲区"""
        # 将经验以元组形式存储
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """从缓冲区随机采样一个批次的经验"""
        # random.sample() 从序列中无放回地随机选择 batch_size 个元素
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*random.sample(self.buffer, batch_size))
        
        # 将采样到的经验转换为 PyTorch 张量
        # action 和 done 需要 long 类型，reward 需要 float 类型
        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float32)
        action_batch = torch.tensor(np.array(action_batch), dtype=torch.int64)
        reward_batch = torch.tensor(np.array(reward_batch), dtype=torch.float32)
        next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float32)
        done_batch = torch.tensor(np.array(done_batch), dtype=torch.float32) # 布尔值转float

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        """返回缓冲区当前的经验数量"""
        return len(self.buffer)

# --- DQN Agent 类 ---
class DQNAgent:
    def __init__(
        self, 
        observation_dim, 
        action_dim, 
        device="cpu",

        # DQN 参数
        batch_size=64,
        lr=0.0005, # 学习率
        gamma=0.99, # 折扣因子
        epsilon_start=1.0, # 初始探索率
        epsilon_end=0.01, # 最终探索率
        epsilon_decay=0.001, # 探索率衰减速度
        target_update_freq=100, # 目标网络更新频率 (每多少个回合更新一次)
        max_episodes=1000, # 最大训练回合数
        max_steps_per_episode=1000, # 每回合最大步数
        replay_buffer_capacity=100_000, # 回放缓冲区容量
        full_score=200, # 完全得分 (LunarLander-v3 的目标是 200 分)
        show_delay=0.05 # 渲染时的延迟，单位秒
    ):
        self.action_dim = action_dim
        self.device = device # 运行设备 (CPU 或 GPU)

        # DQN 参数
        self.batch_size = batch_size
        self.gamma = gamma # 折扣因子
        self.epsilon_start = epsilon_start # 初始探索率
        self.epsilon_end = epsilon_end # 最终探索率
        self.epsilon_decay = epsilon_decay # 探索率衰减速度
        self.target_update_freq = target_update_freq # 目标网络更新频率 (每多少个回合更新一次)
        self.max_episodes = max_episodes # 最大训练回合数
        self.max_steps_per_episode = max_steps_per_episode # 每回合最大步数
        self.full_score = full_score
        self.show_delay = show_delay # 渲染时的延迟，单位秒

        # QNetwork = QLinear
        QNetwork = QResNet # 使用残差网络
        # 当前网络 (Current Network)
        self.current_net = QNetwork(observation_dim, action_dim).to(device)
        # 目标网络 (Target Network) - 与当前网络结构相同
        self.target_net = QNetwork(observation_dim, action_dim).to(device)
        # 初始化时，将目标网络的参数复制为当前网络的参数
        self.target_net.load_state_dict(self.current_net.state_dict())
        self.target_net.eval() # 目标网络只用于评估，不进行训练，因此设置为评估模式

        # 优化器
        self.optimizer = optim.Adam(self.current_net.parameters(), lr=lr) # Adam 优化器
        # 损失函数
        self.criterion = nn.MSELoss() # 均方误差损失

        # 回放缓冲区
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity) # 缓冲区容量


    def select_action(self, state, episode) -> int:
        """
        根据 ε-greedy 策略选择动作。
        state: 当前观测值 (NumPy 数组)
        episode: 当前回合数，用于计算 epsilon

        return: 选择的动作索引 (int)
        """
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-self.epsilon_decay * episode)
        
        if random.random() < epsilon:
            return random.randrange(self.action_dim) # 探索：随机动作
        else:
            # 利用：选择 Q 值最高的动作
            # 将 NumPy 数组转换为 PyTorch 张量，并移动到设备
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                # shape = (1, observation_dim)
            # 禁用梯度计算，因为这里只是前向传播，不需要反向传播
            with torch.no_grad():
                q_values = self.current_net(state_tensor)
            return q_values.argmax(1).item() # 返回 Q 值最高的动作索引

    def update_current_network(self):
        """
        DQN 学习步骤：从回放缓冲区采样，计算损失，更新当前网络。
        """
        # 如果缓冲区中的经验不足以构成一个批次，则不学习
        if len(self.replay_buffer) < self.batch_size:
            return

        # 从缓冲区采样一个批次
        states, actions, rewards, next_states, terminateds = self.replay_buffer.sample(self.batch_size)
        
        # 将数据移动到指定设备 (CPU/GPU)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        terminateds = terminateds.to(self.device)

        # 计算当前状态的 Q 值 (Q(s, a))
        # current_net(states) 会返回所有动作的 Q 值，然后用 actions 索引出实际执行动作的 Q 值
        # q_values = self.current_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        all_q_values = self.current_net(states) # (batch_size, action_dim)
        q_values = all_q_values.gather(1, actions.unsqueeze(1)) # (batch_size, 1)
        q_values = q_values.squeeze(1) # 去掉多余的维度，变为 (batch_size,)

        # 计算目标 Q 值 (r + gamma * max Q(s', a'))
        with torch.no_grad(): # 目标网络不需要梯度
            # max_next_q_values = self.target_net(next_states).max(1)[0]
            all_next_q_values = self.target_net(next_states) # (batch_size, action_dim)
            max_next_q_values = all_next_q_values.max(1) # 返回 (max_q_value, indices) 的元组
            max_next_q_values = max_next_q_values[0] # 取出最大 Q 值 (batch_size,)
        new_q_values = rewards + self.gamma * max_next_q_values * (1 - terminateds)
        # (1 - terminateds) 用于处理终止状态：如果回合结束 (terminated=True)，则 max_next_q 为 0

        # 计算损失
        loss = self.criterion(q_values, new_q_values)

        # 反向传播和优化
        self.optimizer.zero_grad() # 清零梯度
        loss.backward() # 反向传播，计算梯度
        self.optimizer.step() # 更新网络参数
    
    def train(self, train_env):
        """
        训练 DQN 智能体。
        :param train_env: gym 环境
        """
        rewards_per_episode = []

        for episode in range(self.max_episodes):
            observation, info = train_env.reset()
            current_state = observation
            done = False
            current_episode_reward = 0.0

            for step in range(self.max_steps_per_episode):
                action = agent.select_action(current_state, episode) # 智能体选择动作

                new_observation, reward, terminated, truncated, info = train_env.step(action)
                new_state = new_observation
                current_episode_reward += float(reward)

                # 将经验存储到回放缓冲区
                agent.replay_buffer.push(current_state, action, reward, new_state, done)

                # 每步都尝试学习（如果缓冲区有足够数据）
                agent.update_current_network()

                current_state = new_state
                if terminated or truncated:
                    done = True
                    break
            
            rewards_per_episode.append(current_episode_reward)

            # 定期更新目标网络
            if (episode + 1) % agent.target_update_freq == 0:
                agent.target_net.load_state_dict(agent.current_net.state_dict())
                print(f"--- 目标网络已更新 (回合 {episode + 1}) ---")

            # 打印训练进度
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_per_episode[-100:])
                print(f"回合 {episode + 1}/{self.max_episodes}, 平均奖励 (最近100回合): {avg_reward:.2f}, "
                      f"Epsilon: {agent.epsilon_end + (agent.epsilon_start - agent.epsilon_end) * np.exp(-agent.epsilon_decay * episode):.4f}")

                # 达到一定平均奖励后停止训练 (可选)
                # LunarLander 的目标是 200 分以上
                if avg_reward >= self.full_score:
                    print(f"智能体已学会着陆！在 {episode + 1} 回合平均奖励 {avg_reward:.2f} 达到满分 {self.full_score} 。")
                    break
        
        print("\nDQN 训练完成！")
        self.draw_train_score(rewards_per_episode)  # 绘制训练过程中每个回合的总奖励
        return rewards_per_episode

    def show_train_results(self, env):
        assert self.target_net is not None, "请先训练智能体！"

        observation, info = env.reset()
        current_state = observation
        done = False
        total_eval_reward = 0
        eval_steps = 0

        while not done and eval_steps < self.max_steps_per_episode: # 评估也有限制步数
            next_q_values = self.target_net(torch.tensor(current_state, dtype=torch.float32).unsqueeze(0).to(self.device))
            action = next_q_values.argmax(1).item()
            new_observation, reward, terminated, truncated, info = env.step(action)
            env.render() # 渲染当前帧
            new_state = new_observation
            total_eval_reward += reward
            eval_steps += 1
            current_state = new_state
            if terminated or truncated:
                done = True
            time.sleep(self.show_delay) # 让动画慢一点，方便观察
        env.close()
        print(f"评估完成！总评估奖励: {total_eval_reward} (运行了 {eval_steps} 步).")
    
    def draw_train_score(self, rewards_per_episode):
        """
        绘制训练过程中每个回合的总奖励。
        :param rewards_per_episode: 每个回合的总奖励列表
        """
        plt.figure(figsize=(12, 6))
        plt.plot(rewards_per_episode)
        plt.title('DQN 训练过程中每回合的总奖励')
        plt.xlabel('回合数')
        plt.ylabel('总奖励')
        plt.grid(True)
        plt.show()

        # 绘制移动平均
        N = 100 # 移动平均窗口大小
        if len(rewards_per_episode) >= N:
            running_avg_rewards = np.convolve(rewards_per_episode, np.ones(N)/N, mode='valid')
            plt.figure(figsize=(12, 6))
            plt.plot(running_avg_rewards)
            plt.title(f'DQN 训练过程中每回合总奖励的 {N} 回合移动平均')
            plt.xlabel('回合数')
            plt.ylabel('平均总奖励')
            plt.grid(True)
            plt.show()
    
    def save_model(self, filename):
        """
        保存当前网络模型到文件。
        :param filename: 保存的文件名。
        """
        assert self.current_net is not None, "请先训练智能体！"
        torch.save(self.current_net.state_dict(), filename)
        print(f"模型已保存到 {filename}")
    
    def load_model(self, filename):
        """
        从文件加载当前网络模型。
        :param filename: 要加载的文件名。
        """
        self.current_net.load_state_dict(torch.load(filename, map_location=self.device))
        self.current_net.eval()

# --- 主训练函数 ---
def train_dqn(agent, train_env, eval_env, n_episodes=1000, max_steps_per_episode=1000):
    rewards_per_episode = []

    for episode in range(n_episodes):
        observation, info = train_env.reset()
        current_state = observation
        done = False
        current_episode_reward = 0

        for step in range(max_steps_per_episode):
            action = agent.select_action(current_state, episode) # 智能体选择动作

            new_observation, reward, terminated, truncated, info = train_env.step(action)
            new_state = new_observation
            current_episode_reward += reward

            # 将经验存储到回放缓冲区
            agent.replay_buffer.push(current_state, action, reward, new_state, terminated or truncated)

            # 每步都尝试学习（如果缓冲区有足够数据）
            agent.update_target_network()

            current_state = new_state
            if terminated or truncated:
                done = True
                break
        
        rewards_per_episode.append(current_episode_reward)

        # 定期更新目标网络
        if (episode + 1) % agent.target_update_freq == 0:
            agent.target_net.load_state_dict(agent.current_net.state_dict())
            print(f"--- 目标网络已更新 (回合 {episode + 1}) ---")

        # 打印训练进度
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"回合 {episode + 1}/{n_episodes}, 平均奖励 (最近100回合): {avg_reward:.2f}, Epsilon: {agent.epsilon_end + (agent.epsilon_start - agent.epsilon_end) * np.exp(-agent.epsilon_decay * episode):.4f}")

            # 达到一定平均奖励后停止训练 (可选)
            # LunarLander 的目标是 200 分以上
            if avg_reward >= 200:
                print(f"智能体已学会着陆！在 {episode + 1} 回合达到平均奖励 {avg_reward:.2f}")
                break
    
    print("\nDQN 训练完成！")
    return rewards_per_episode

def evaluate_dqn(agent, eval_env, num_eval_episodes=5, max_steps_per_episode=1000):
    """
    评估训练好的 DQN 智能体，并进行可视化。
    """
    print("\n开始评估训练好的智能体 (可视化)...")
    total_rewards = []

    for episode in range(num_eval_episodes):
        observation, info = eval_env.reset()
        current_state = observation
        done = False
        episode_reward = 0
        eval_steps = 0

        while not done and eval_steps < max_steps_per_episode:
            # 评估时只进行利用，不探索
            state_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action = agent.current_net(state_tensor).argmax(1).item()

            new_observation, reward, terminated, truncated, info = eval_env.step(action)
            eval_env.render() # 渲染当前帧
            new_state = new_observation
            episode_reward += reward
            eval_steps += 1
            current_state = new_state
            if terminated or truncated:
                done = True
            time.sleep(0.05) # 动画延迟

        total_rewards.append(episode_reward)
        print(f"评估回合 {episode + 1}/{num_eval_episodes}, 总奖励: {episode_reward:.2f} (运行了 {eval_steps} 步).")
    
    eval_env.close()
    print(f"评估完成！平均评估奖励: {np.mean(total_rewards):.2f}.")


# --- 5. 主程序运行 ---
if __name__ == "__main__":
    # 检查是否有 GPU 可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 创建训练环境 (不渲染)
    train_env = gym.make("LunarLander-v3")
    # 创建评估环境 (渲染)
    eval_env = gym.make("LunarLander-v3", render_mode="human")

    # 获取环境信息
    # observation_dim = train_env.observation_space.shape[0] # LunarLander 有 8 个观测值
    # action_dim = train_env.action_space.n # LunarLander 有 4 个动作
    observation_dim = 8
    action_dim = 4

    print(f"观测维度: {observation_dim}")
    print(f"动作数量: {action_dim}")

    # 初始化 DQN Agent
    agent = DQNAgent(observation_dim, action_dim, device)

    # 训练 DQN
    rewards_history = agent.train(train_env)

    # 展示训练结果
    agent.show_train_results(eval_env)

    # 关闭环境
    train_env.close()
    eval_env.close()
    print("\n所有环境已关闭。程序执行完毕。")