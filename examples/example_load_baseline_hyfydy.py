# This example requires the installation of sconegym:
# https://github.com/tgeijten/sconegym

import gym
import sconegym  # noqa

import deprl

# create the sconegym env
env = gym.make("sconewalk_h0918-v1")

# choose one of these for another baseline

# env = gym.make("sconewalk_h1622-v1")
# env = gym.make("sconewalk_h2190-v1")
# env = gym.make("sconerun_h0918-v1")
# env = gym.make("sconerun_h1622-v1")
# env = gym.make("sconerun_h2190-v1")

policy = deprl.load_baseline(env)

env.seed(0)
for ep in range(5):
    if ep % 1 == 0:
        env.store_next_episode()  # Store results of every Nth episode

    ep_steps = 0
    ep_tot_reward = 0
    state = env.reset()

    while True:
        # samples random action
        action = policy(state)
        # applies action and advances environment by one step
        state, reward, done, info = env.step(action)

        ep_steps += 1
        ep_tot_reward += reward

        # check if done
        if done or (ep_steps >= 1000):
            print(
                f"Episode {ep} ending; steps={ep_steps}; reward={ep_tot_reward:0.3f}; \
                com={env.model.com_pos()}"
            )
            env.write_now()
            env.reset()
            break

env.close()
