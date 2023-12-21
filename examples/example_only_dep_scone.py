import gym
import sconegym  # noqa

from deprl import env_wrappers
from deprl.dep_controller import DEP

# create the sconegym env
env = gym.make("sconewalk_h2190-v1")

# apply wrapper to environment
env = env_wrappers.SconeWrapper(env)

# create DEP, parameters are loaded from default path
dep = DEP()

# give DEP obs and action space to create right dimensions
dep.initialize(env.observation_space, env.action_space)

env.seed(0)

for ep in range(5):
    if ep % 1 == 0:
        env.store_next_episode()  # Store results of every Nth episode

    ep_steps = 0
    ep_tot_reward = 0
    state = env.reset()

    while True:
        # samples random action
        action = dep.step(env.muscle_lengths())[0, :]
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
