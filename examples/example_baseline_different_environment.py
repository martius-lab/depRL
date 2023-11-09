import gym
import sconegym  # noqa

import deprl

# create the sconegym env
env = gym.make("sconewalk_h1622-v0")

# pretrained baselines
policy = deprl.load_baseline(env)

# example of loading baseline but executing with another model
# NOTE they need to have an identical observation space
env = gym.make("sconewalk_h1622-v1")

env.seed(0)
for ep in range(10):
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
