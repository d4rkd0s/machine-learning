import gym

# 6_openai_cartpole_test.py
# Start of the OpenAI gym tests, here I simply have gotten
# the cartpole running and am planning to write code to complete it.

env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()