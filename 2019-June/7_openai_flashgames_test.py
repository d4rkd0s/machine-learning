import gym
import universe

# 7_openai_flashgames_test.py
# Same as 6, here I simply have gotten flash games
# running and am planning to write code to complete it.

env = gym.make('flashgames.CoasterRacer-v0')
observation_n = env.reset()

while True:
    action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]
    observation_n, reward_n, done_n, info = env.step(action_n)
    env.render()
