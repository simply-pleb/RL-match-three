import torch
import numpy as np
from gym_match3.envs import Match3Env
from improved_a3c import A3CModel

input_shape = (9, 9, 4)
num_actions = 144

# Load trained model
model = A3CModel(input_shape, num_actions)
model.load_state_dict(torch.load("a3c_match3_trained.pt", map_location="cpu"))
model.eval()

env = Match3Env()
state = env.reset()
done = False
total_reward = 0
steps = 0

while not done and steps < 100:  # Limit steps for demo
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(state_tensor)
        probs = torch.softmax(logits, dim=1)
        action = torch.argmax(probs, dim=1).item()
    next_state, reward, done, _ = env.step(action)
    if isinstance(reward, list) and len(reward) == 3:
        scalar_reward = float(reward[1])
    else:
        scalar_reward = 0.0
    total_reward += scalar_reward
    state = next_state
    steps += 1
    print(f"Step {steps}: Action {action}, Reward {scalar_reward}, Done {done}")

print(f"Total reward: {total_reward}")