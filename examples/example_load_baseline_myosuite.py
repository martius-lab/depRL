# This example requires the installation of myosuite
# pip install myosuite

import gym
import myosuite  # noqa

import deprl

# create the sconegym env
env = gym.make("myoLegChaseTagP1-v0")

policy = deprl.load_baseline(env)

env.seed(0)
for ep in range(5):
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
        env.render()

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
