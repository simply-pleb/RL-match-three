import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributions import Categorical
from gym_match3.envs import Match3Env
import matplotlib.pyplot as plt


# A3C Model
class A3CModel(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(A3CModel, self).__init__()
        self.input_shape = input_shape  # (9, 9, 7)
        self.num_actions = num_actions  # 144

        # Depthwise separable convolution
        self.depthwise = nn.Conv2d(
            input_shape[2],
            input_shape[2],
            kernel_size=3,
            padding=1,
            groups=input_shape[2],
        )
        self.pointwise = nn.Conv2d(input_shape[2], 32, kernel_size=1)

        # Output size after convolution: 32 * height * width
        conv_output_size = 32 * input_shape[0] * input_shape[1]

        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc_actor = nn.Linear(128, num_actions)  # Policy head
        self.fc_critic = nn.Linear(128, 1)  # Value head

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Reshape from (batch, h, w, c) to (batch, c, h, w)
        x = F.relu(self.pointwise(F.relu(self.depthwise(x))))
        x = x.contiguous()  # Ensure tensor is contiguous
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        policy_logits = self.fc_actor(x)
        value = self.fc_critic(x)
        return policy_logits, value


# Compute n-step returns
def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


# Worker function
def worker(
    worker_id,
    global_model,
    optimizer,
    lock,
    reward_queue,
    num_episodes,
    input_shape,
    num_actions,
):
    local_model = A3CModel(input_shape, num_actions)
    local_model.load_state_dict(global_model.state_dict())

    env = Match3Env()
    state = env.reset()
    episode = 0
    episode_reward = 0.0  # Initialize as float
    n_steps = 20
    step_count = 0  # For debugging reward structure

    while episode < num_episodes:
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0

        for _ in range(n_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            policy_logits, value = local_model(state_tensor)
            probs = F.softmax(policy_logits, dim=1)
            m = Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            entropy += m.entropy().mean()

            next_state, reward, done, _ = env.step(action.item())
            # Debug reward structure for first 5 steps
            if step_count < 5:
                print(
                    f"Worker {worker_id}, Step {step_count}: Reward = {reward}, Type = {type(reward)}"
                )
                step_count += 1
            # Extract scalar reward (reward_value) from list
            if not isinstance(reward, list) or len(reward) != 3:
                raise ValueError(f"Unexpected reward format: {reward}")
            scalar_reward = float(reward[1])  # Use reward[1] as the scalar reward
            episode_reward += scalar_reward

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor([scalar_reward]))
            masks.append(torch.FloatTensor([1 - done]))

            state = next_state

            if done:
                reward_queue.put(episode_reward)
                episode_reward = 0.0
                state = env.reset()
                episode += 1
                if episode >= num_episodes:
                    break

        # Compute next value for n-step return
        next_state_tensor = torch.FloatTensor(state).unsqueeze(0)
        # Ensure consistent tuple return
        policy_logits, next_value = (
            local_model(next_state_tensor)
            if not done
            else (None, torch.FloatTensor([0]))
        )

        returns = compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        with lock:
            for gp, lp in zip(global_model.parameters(), local_model.parameters()):
                if lp.grad is not None:
                    gp.grad = lp.grad.clone()
            optimizer.step()
        local_model.load_state_dict(global_model.state_dict())


# Main function
if __name__ == "__main__":
    mp.set_start_method("spawn")

    input_shape = (9, 9, 7)  # 9x9 board with up to 7 shapes
    num_actions = 144  # 72 horizontal + 72 vertical swaps
    num_workers = 4
    episodes_per_worker = 100

    global_model = A3CModel(input_shape, num_actions)
    global_model.share_memory()
    optimizer = optim.Adam(global_model.parameters(), lr=1e-4)
    lock = mp.Lock()
    reward_queue = mp.Queue()

    processes = []
    for worker_id in range(num_workers):
        p = mp.Process(
            target=worker,
            args=(
                worker_id,
                global_model,
                optimizer,
                lock,
                reward_queue,
                episodes_per_worker,
                input_shape,
                num_actions,
            ),
        )
        p.start()
        processes.append(p)

    # Collect and plot rewards
    episode_rewards = []
    while len(episode_rewards) < num_workers * episodes_per_worker:
        try:
            reward = reward_queue.get(timeout=10)
            episode_rewards.append(reward)
            if len(episode_rewards) % 100 == 0:
                print(f"Collected {len(episode_rewards)} episodes")
        except Exception as e:
            print(e)
            break

    for p in processes:
        p.join()
    
    import pickle
    with open("episode_rewards.pkl", mode="wb") as f:
        pickle.dump(episode_rewards, f)

    # plt.plot(episode_rewards)
    # plt.xlabel("Episode")
    # plt.ylabel("Reward")
    # plt.title("A3C Training on Match-3 Environment")
    # plt.show()
