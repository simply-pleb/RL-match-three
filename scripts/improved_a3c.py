import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributions import Categorical
from gym_match3.envs import Match3Env
import time

# Improved A3C Model with orthogonal initialization and BatchNorm
# class A3CModel(nn.Module):
#     def __init__(self, input_shape, num_actions):
#         super(A3CModel, self).__init__()
#         C, H, W = input_shape[2], input_shape[0], input_shape[1]
#         self.depthwise = nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C)
#         self.bn1 = nn.BatchNorm2d(C)
#         self.pointwise = nn.Conv2d(C, 32, kernel_size=1)
#         self.bn2 = nn.BatchNorm2d(32)
#         conv_output_size = 32 * H * W
#         self.fc1 = nn.Linear(conv_output_size, 128)
#         self.fc_actor = nn.Linear(128, num_actions)
#         self.fc_critic = nn.Linear(128, 1)
#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#                 nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = x.permute(0, 3, 1, 2)
#         x = F.relu(self.bn1(self.depthwise(x)))
#         x = F.relu(self.bn2(self.pointwise(x)))
#         x = x.contiguous().view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         return self.fc_actor(x), self.fc_critic(x)

class A3CModel(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(A3CModel, self).__init__()
        C, H, W = input_shape[2], input_shape[0], input_shape[1]
        self.depthwise = nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C)
        conv_output_size = C * H * W
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc_actor = nn.Linear(128, num_actions)
        self.fc_critic = nn.Linear(128, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = F.relu(self.depthwise(x))
        x = x.contiguous().view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc_actor(x), self.fc_critic(x)


def compute_returns_and_advantages(next_value, rewards, masks, values, gamma=0.99, tau=1.0):
    R = next_value
    returns = []
    gae = 0
    advantages = []
    values = values + [next_value]
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        advantages.insert(0, gae)
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns, advantages

def worker(worker_id, global_model, optimizer, lock, reward_queue, num_episodes, input_shape, num_actions):
    local_model = A3CModel(input_shape, num_actions)
    local_model.load_state_dict(global_model.state_dict())
    number_of_steps = 100  # Rollout length (set short episodes for debugging)
    env = Match3Env(rollout_len=number_of_steps)  
    episode = 0

    while episode < num_episodes:
        try:
            state = env.reset()
            done = False
            log_probs, values, rewards, masks = [], [], [], []
            episode_reward = 0
            step_count = 0

            while not done and len(rewards) < number_of_steps:  # Short rollout for debugging
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                logits, value = local_model(state_tensor)
                probs = F.softmax(logits, dim=1)
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy().mean()

                next_state, reward, done, _ = env.step(action.item())
                # Debug print
                print(f"[Worker {worker_id}] Step {step_count} | Action: {action.item()} | Reward: {reward}")

                if not isinstance(reward, list) or len(reward) != 3:
                    print(f"[Worker {worker_id}] Unexpected reward format: {reward}")
                    scalar_reward = 0.0
                else:
                    scalar_reward = float(reward[1])
                episode_reward += scalar_reward

                log_probs.append(log_prob)
                values.append(value.squeeze(1))
                rewards.append(torch.tensor(scalar_reward))
                masks.append(torch.tensor(1 - done, dtype=torch.float32))

                state = next_state
                step_count += 1

            # Compute next value
            next_val = torch.zeros(1)
            if not done:
                _, next_val = local_model(torch.FloatTensor(state).unsqueeze(0))
                next_val = next_val.squeeze(1)

            returns, advantages = compute_returns_and_advantages(
                next_val, rewards, masks, values, gamma=0.99, tau=0.95
            )
            returns = torch.stack(returns).detach()
            advantages = torch.stack(advantages).detach()
            log_probs = torch.stack(log_probs)
            values = torch.stack(values)

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Losses
            actor_loss = -(log_probs * advantages).mean()
            critic_loss = F.mse_loss(values, returns)
            loss = actor_loss + 0.5 * critic_loss  # entropy not used for simplicity

            # Update global model
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(global_model.parameters(), 0.5)
            with lock:
                for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                    if local_param.grad is not None:
                        global_param._grad = local_param.grad
                optimizer.step()
                local_model.load_state_dict(global_model.state_dict())

            reward_queue.put(episode_reward)
            print(f"[Worker {worker_id}] Episode {episode} finished. Total reward: {episode_reward}")
            episode += 1

        except Exception as e:
            print(f"[Worker {worker_id}] Exception: {e}")
            reward_queue.put(0)
            episode += 1

if __name__ == "__main__":
    start_time = time.time()
    mp.set_start_method("spawn")
    input_shape = (9, 9, 4)
    num_actions = 144
    num_workers = 16  # Increase workers for faster training
    episodes_per_worker = 100  # Increase episodes for real training

    global_model = A3CModel(input_shape, num_actions)
    global_model.share_memory()
    optimizer = optim.RMSprop(global_model.parameters(), lr=7e-4, alpha=0.99, eps=1e-5)
    lock = mp.Lock()
    reward_queue = mp.Queue()

    processes = []
    for wid in range(num_workers):
        p = mp.Process(
            target=worker,
            args=(wid, global_model, optimizer, lock, reward_queue,
                  episodes_per_worker, input_shape, num_actions),
        )
        p.start()
        processes.append(p)

    episode_rewards = []
    while len(episode_rewards) < num_workers * episodes_per_worker:
        try:
            reward = reward_queue.get(timeout=60)
            if len(episode_rewards) % 100 == 0:
                print(f"[Main] Collected {len(episode_rewards)} rewards")
            episode_rewards.append(reward)
        except Exception as e:
            print(f"[Main] Exception while collecting rewards: {e}")
            break

    for p in processes:
        p.join()

    # Calculate and print training time
    training_time = time.time() - start_time
    print(f"[Main] Training time: {training_time / 60} minutes ({training_time} seconds)")
    # Save rewards for inspection
    import pickle
    with open("episode_rewards_a3c_paper.pkl", "wb") as f:
        pickle.dump(episode_rewards, f)
    print("[Main] Training finished. Rewards saved to episode_rewards_a3c_paper.pkl")

    # Save the trained model
    torch.save(global_model.state_dict(), "paper_a3c_match3_trained.pt")
    print("[Main] Trained model saved to paper_a3c_match3_trained.pt")
