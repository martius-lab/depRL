import gym
import sconegym
import deprl

# create the sconegym env
env = gym.make("sconegait2d-v0")
env = deprl.apply_wrapper(env)
policy = deprl.load('./sconerun/', env)


for episode in range(100):
    # store the results of every 10th episode
    # storing results is slow, and should only be done sparsely
    # stored results can be analyzed in SCONE Studio
    if episode % 5 == 0:
        env.store_next_episode()

    episode_steps = 0
    total_reward = 0
    state = env.reset()

    while True:
        # samples random action
        action = policy(state)

        # applies action and advances environment by one step
        next_state, reward, done, info = env.step(action)

        episode_steps += 1
        total_reward += reward

        # to render results, open a .sto file in SCONE Studio
        #env.render()
        state = next_state

        # check if done
        if done or (episode_steps >= 1000):
            print(f'Episode {episode} finished; steps={episode_steps}; reward={total_reward:0.3f}')
            env.write_now()
            episode += 1
            break
        
env.close()
