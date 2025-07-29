import gymnasium as gym
import pygame
import time
from gymnasium.utils.play import play

# 1. 创建 LunarLander-v3 环境
# 使用 render_mode="human" 来显示游戏窗口
env = gym.make("LunarLander-v3", render_mode="rgb_array")

# 2. 定义键盘按键到动作的映射
# LunarLander-v3 的动作空间是离散的，有 4 个动作：
# 0: 不操作 (do nothing)
# 1: 主引擎点火 (fire main engine) - 向下推力
# 2: 左侧引擎点火 (fire left engine) - 向右推力
# 3: 右侧引擎点火 (fire right engine) - 向左推力
key_to_action = {
    # 动作 0: 不操作 - 可以不映射任何键，或映射一个不常用的键
    # 动作 2: 主引擎点火 (向上飞 / 减缓下降)
    (pygame.K_UP,): 2,
    # 动作 3: 左侧引擎点火 (向右移动 / 纠正向左的旋转)
    (pygame.K_LEFT,): 3,
    # 动作 1: 右侧引擎点火 (向左移动 / 纠正向右的旋转)
    (pygame.K_RIGHT,): 1,
}

# play 函数的 noop 参数：当没有按键按下时执行的默认动作
# 对于 LunarLander，默认“不操作”是动作 0，所以设置为 0
print("---")
print("使用箭头键 (上、左、右) 来控制登月器。")
print("按 ESC 键或关闭窗口退出。")
print("---")

# 3. 调用 play 函数开始手动游戏
# zoom 参数可以放大显示窗口，方便观察
# fps 是帧率，可以控制游戏的运行速度
play(env, keys_to_action=key_to_action, noop=0, zoom=1.5, fps=20) 

# 4. 游戏结束后自动关闭环境
env.close()

print("\n手动操控 LunarLander 结束。")