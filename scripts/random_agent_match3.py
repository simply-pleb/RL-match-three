import pickle
from gym_match3.envs import Match3Env
import numpy as np

# Match A3C settings
input_shape = (8, 8, 7)
num_actions = 112
num_workers = 16
episodes_per_worker = 100
number_of_steps = 100  # Rollout length

def run_random_agent(worker_id, reward_queue, num_episodes, num_actions):
    env = Match3Env(rollout_len=number_of_steps)
    episode = 0
    while episode < num_episodes:
        state = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        while not done and step_count < number_of_steps:
            action = np.random.randint(num_actions)
            next_state, reward, done, _ = env.step(action)
            # Reward extraction (same as A3C)
            if not isinstance(reward, list) or len(reward) != 3:
                scalar_reward = 0.0
            else:
                scalar_reward = float(reward[1])
            episode_reward += scalar_reward
            state = next_state
            step_count += 1
        reward_queue.append(episode_reward)
        print(f"[Random Worker {worker_id}] Episode {episode} finished. Total reward: {episode_reward}")
        episode += 1

if __name__ == "__main__":
    import multiprocessing as mp
    import time

    start_time = time.time()
    reward_queue = mp.Manager().list()
    processes = []
    for wid in range(num_workers):
        p = mp.Process(
            target=run_random_agent,
            args=(wid, reward_queue, episodes_per_worker, num_actions),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    episode_rewards = list(reward_queue)
    training_time = time.time() - start_time
    print(f"[Main] Random agent time: {training_time / 60} minutes ({training_time} seconds)")

    # Save rewards for inspection
    with open("episode_rewards_random.pkl", "wb") as f:
        pickle.dump(episode_rewards, f)
    print("[Main] Random agent finished. Rewards saved to episode_rewards_random.pkl")