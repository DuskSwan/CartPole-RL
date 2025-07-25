import time
import math
import numpy as np

import gymnasium as gym
from gymnasium.utils.play import play
import pygame  # 导入 Pygame 库
import pygame.constants as pygame_keys # 导入 Pygame 键盘常量

def play_cartpole():
    # 创建环境，渲染模式为 RGB 数组
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    obs, info = env.reset(seed=42)
    score = 0

    # 初始化 pygame
    pygame.init()
    # 获取第一个渲染帧的尺寸
    frame: np.ndarray = env.render()
    height, width, _ = frame.shape
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('CartPole Manual Control')
    clock = pygame.time.Clock()
    fps = 30  # 设置帧率

    max_degrees = 45  # 设置杆子倾斜角度限制
    env.unwrapped.theta_threshold_radians = max_degrees * (math.pi / 180)
    print(f"杆子倾斜角度限制已设置为 ±{max_degrees} 度 ({env.unwrapped.theta_threshold_radians:.4f} 弧度)")

    game_over = False
    turn = 0  # 用于计数回合数
    while True:
        turn += 1
        if game_over:
            # 如果游戏结束，等待用户按下空格键重新开始
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    # 按空格键重置游戏
                    obs, info = env.reset()
                    game_over = False
                    score = 0
                    turn = 0
        
        else:
            action = 0 if obs[2] < 0 else 1
            pygame.event.pump() # 更新 Pygame 事件队列，如果不调用pygame.event.get()就需要这个
            keys = pygame.key.get_pressed()  # 获取当前按键状态
            if keys[pygame_keys.K_LEFT]:
                action = 0
                key = 'LEFT'
            elif keys[pygame_keys.K_RIGHT]:
                action = 1
                key = 'RIGHT'
            else:
                key = 'NONE'

            # 执行动作
            print(f"回合 {turn}: 输入{key} 执行动作 {action}")
            obs, reward, terminated, truncated, info = env.step(action)
            score += reward

            # 渲染环境并转换为 pygame Surface
            frame: np.ndarray = env.render() # np.array: (height, width, 3)
            surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1)) # (width, height, 3) -> (width, height)
            screen.blit(surf, (0, 0)) # blit 是 Pygame 的 “图像贴到画布” 操作。这里讲 surf（当前帧画面）贴到 screen 画布的左上角坐标 (0, 0) 上。
            pygame.display.flip()

            # 如果回合结束，进入暂停（游戏结束）状态
            if (terminated or truncated):
                game_over = True
                # 显示 Game Over 提示
                font = pygame.font.SysFont(None, 48)
                text = font.render('Game Over! Press SPACE to restart.', True, (255, 0, 0))
                text_rect = text.get_rect(center=(width // 2, height // 2))
                screen.blit(text, text_rect)
                pygame.display.flip()
                print(f"回合 {turn} 结束，得分: {score}")

        # 控制帧率
        clock.tick(fps)

def test():
    env = gym.make("CartPole-v1", render_mode="human")
    observation, info = env.reset()

    for _ in range(600): # 运行600帧
        action = env.action_space.sample() # 随机动作
        observation, reward, terminated, truncated, info = env.step(action)
        env.render() # 每次都渲染
        if terminated or truncated:
            break
        time.sleep(0.01) # 短暂暂停，观察效果

    env.close()

if __name__ == "__main__":
    play_cartpole()
    # test()