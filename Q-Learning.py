import time, sys, math
from loguru import logger

import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import ActionWrapper, ObservationWrapper

class QLearningAgent:
    def __init__(
            self,
            learning_rate = 0.1,      # α, 学习率
            discount_factor = 0.99,    # γ, 折扣因子，未来奖励的重要性
            epsilon_start = 1.0,       # ε, 初始探索率 (100% 探索)
            epsilon_end = 0.01,        # 最终探索率 (1% 探索)
            epsilon_decay_rate = 0.001, # 探索率衰减速度 (每回合减少多少)
            print_interval = 100,   # 每多少回合打印一次训练进度
            show_delay = 0.05,      # 渲染时的延迟，单位秒
            n_episodes = 10000,         # 训练回合数，也即总共进行多少局游戏
            max_steps_per_episode = 300, # 每个回合的最大步数 (CartPole-v1 默认是 500，这里可以设小一点方便快速测试)
        ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_rate = epsilon_decay_rate
        self.n_episodes = n_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.print_interval = print_interval
        self.show_delay = show_delay
        self.q_table = None  # Q 表将在训练时初始化
    
    def train(self, env):
        """
        训练 Q-Learning 智能体。
        :param env: 包装过的gym环境，observation已经离散化。
        """
        n_observation = env.observation_space.n  # 观测空间维度
        n_actions = env.action_space.n  # 动作空间维度
        q_table = np.zeros((n_observation, n_actions))  # 初始化 Q 表
        rewards_per_episode = []
        for episode in range(self.n_episodes):
            # 计算当前回合的 epsilon，也即探索率
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-self.epsilon_decay_rate * episode)

            # 重置环境，获取初始观测值
            observation, info = env.reset()

            done = False
            current_episode_reward = 0
            for step in range(self.max_steps_per_episode):
                # ε-greedy 策略：选择动作
                if np.random.rand() < epsilon:
                    action = env.action_space.sample() # 探索：随机选择动作
                else:
                    action = np.argmax(q_table[observation, :]) # 利用：选择 Q 值最高的动作

                # 执行动作，获取新的状态和奖励
                new_observation, reward, terminated, truncated, info = env.step(action)
                current_episode_reward += reward

                # 更新公式
                if terminated or truncated:
                    max_future_q = 0 # 游戏结束时，未来奖励为 0
                    done = True
                else:
                    max_future_q = np.max(q_table[new_observation, :]) # 使用下一个状态的最大 Q 值作为未来奖励
                # Q 值更新公式的核心
                q_table[observation, action] = q_table[observation, action] + self.learning_rate * (
                    reward + self.discount_factor * max_future_q - q_table[observation, action]
                )
                # 更新状态
                observation = new_observation
                if done: break
            
            rewards_per_episode.append(current_episode_reward)

            # 打印训练进度
            gap = self.print_interval
            if (episode + 1) % gap == 0:
                avg_reward = np.mean(rewards_per_episode[-gap:])
                print(f"回合 {episode + 1}/{self.n_episodes}, 平均奖励(最近{gap}回合): {avg_reward:.2f}")

        self.q_table = q_table
        print("\nQ-Learning 训练完成！")
        env.close()
        self.draw_train_score(rewards_per_episode)  # 绘制训练过程中每个回合的总奖励
    
    def draw_train_score(self, rewards_per_episode):
        """
        绘制训练过程中每个回合的总奖励。
        :param rewards_per_episode: 每个回合的总奖励列表。
        """
        plt.plot(rewards_per_episode)
        plt.title("Q-Learning Training Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.show()

    def show_train_results(self, env):
        assert self.q_table is not None, "请先训练智能体！"

        observation, info = env.reset()
        current_state = observation
        done = False
        total_eval_reward = 0
        eval_steps = 0

        while not done and eval_steps < self.max_steps_per_episode: # 评估也有限制步数
            action = np.argmax(self.q_table[current_state, :]) # 利用：根据学习到的 Q 表选择最佳动作
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

class CartPoleWrapper(ObservationWrapper):
    def __init__(self, env, max_degrees=None, num_bins=(2, 2, 6, 3)):
        super().__init__(env)
        self.origin_observation_space = env.observation_space
        # 定义每个维度离散化的“桶”的数量
        self.num_bins = num_bins
        self.init_bins()
        if max_degrees is not None:
            self.set_angle_limit(env, max_degrees)

    def calc_bin_index(self, value, bins):
        """
        计算给定值在指定桶中的索引，从0开始。
        bins 是一个一维数组，表示桶的边界。一个容量为N的桶会有 N-1 个边界。
        """
        if len(bins) == 0: return 0
        return np.digitize(value, bins)

    def init_bins(self):
        self.bins = []
        # 定义每个维度的取值范围（手动指定或根据实际观测范围调整）
        for i, N in enumerate(self.num_bins):
            lower_bound = max(self.origin_observation_space.low[i], -5.0)  # 负无穷被限制为 -5.0
            upper_bound = min(self.origin_observation_space.high[i], 5.0)
            i_index = np.linspace(lower_bound, upper_bound, N + 1)[1:-1]  # 去掉首尾边界
            self.bins.append(i_index)
        # 计算总的离散状态数量
        self.total_discrete_states = math.prod(self.num_bins)
        logger.info(f"离散状态总数: {self.total_discrete_states} (每个维度的桶数量: {self.num_bins})")
        logger.debug(f"每个维度的桶边界: {self.bins}")
        # 更新观测空间
        self.observation_space = gym.spaces.Discrete(self.total_discrete_states)
    
    def discretize_state(self, observation):
        """
        将连续的 CartPole 观测值离散化为一个整数状态。
        """
        index = 0
        for i in range(len(observation)):
            base = math.prod(self.num_bins[:i])
            value = observation[i]
            # 计算离散化后的索引
            index += self.calc_bin_index(value, self.bins[i]) * base
        logger.debug(f"离散化状态: {index} (原始观测值: {observation})")
        return index
    
    def observation(self, obs):
        return self.discretize_state(obs)

    def set_angle_limit(self, env, max_degrees=45):
        env.unwrapped.theta_threshold_radians = max_degrees * (math.pi / 180)
        logger.info(f"杆子倾斜角度限制已设置为 ±{max_degrees} 度 ({env.unwrapped.theta_threshold_radians:.4f} 弧度)")


logger.remove()
logger.add(sys.stderr, level="INFO")
# logger.level("INFO")

'''
Num     Observations     Min     Max
0 cart position        -4.8 4.8
1 cart velocity        -inf inf (often clipped to -3.0 to 3.
2 pole angle           -24 deg to 24 deg (approx -0.418 to 0.418 radians)
3 pole angular velocity -inf inf (often clipped to -3.0 to 3.0 in practice)
'''

max_degrees = 30  # 设置杆子倾斜角度限制
train_env = CartPoleWrapper(gym.make('CartPole-v1'), max_degrees=max_degrees)
agent = QLearningAgent()
agent.train(train_env)  # 训练智能体
show_env = CartPoleWrapper(gym.make('CartPole-v1', render_mode="human"), max_degrees=max_degrees)  
agent.show_train_results(show_env)  # 显示训练结果
