import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import gym
from tqdm import trange
from collections import namedtuple, deque
import matplotlib.pyplot as plt
from matplotlib import animation
import os
import imageio
from PIL import Image, ImageDraw, ImageFont

# Ran on cpu
device = torch.device("cpu")

# Policy network
# Consists of 3 hidden layers with 64 nodes
# Rectified Linear activation function is used

class QNet(nn.Module):
    def __init__(self, states, actions):
        super(QNet, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(states, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, actions)
        )

    def forward(self, x):
        return self.fc(x)

# DQN Class
class DQN():
    def __init__(self, n_states, n_actions, batch_size=64, lr=1e-4, gamma=0.99, mem_size=int(1e5), learn_step=7,
                 tau=1e-3):
        self.n_states = n_states
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_step = learn_step
        self.tau = tau

        # model
        self.net_eval = QNet(n_states, n_actions).to(device)
        self.net_target = QNet(n_states, n_actions).to(device)
        self.optimizer = optim.Adam(self.net_eval.parameters(), lr=lr)
        self.criterion = nn.HuberLoss()
        # self.criterion = nn.MSELoss()

        self.memory = ReplayBuffer(n_actions, mem_size, batch_size)
        self.counter = 0

    def getAction(self, state, epsilon):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.net_eval.eval()
        with torch.no_grad():
            action_values = self.net_eval(state)
        self.net_eval.train()

        # epsilon-greedy
        if random.random() < epsilon:
            action = random.choice(np.arange(self.n_actions))
        else:
            action = np.argmax(action_values.cpu().data.numpy())

        return action

    def save2memory(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.counter += 1
        if self.counter % self.learn_step == 0:
            if len(self.memory) >= self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        q_target = self.net_target(next_states).detach().max(axis=1)[0].unsqueeze(1)
        y_j = rewards + self.gamma * q_target * (1 - dones)
        q_eval = self.net_eval(states).gather(1, actions)

        # loss backpropagation
        loss = self.criterion(q_eval, y_j)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target net
        self.softUpdate()

    def softUpdate(self):
        for eval_param, target_param in zip(self.net_eval.parameters(), self.net_target.parameters()):
            target_param.data.copy_(self.tau * eval_param.data + (1.0 - self.tau) * target_param.data)

# Replay Buffer: Used to store trajectories of experience
class ReplayBuffer:
    def __init__(self, n_actions, memory_size, batch_size):
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def __len__(self):
        return len(self.memory)

    # Append experience to buffer
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    # Grab a random sample from buffer
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        sample_done = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return states, actions, rewards, next_states, sample_done


# Plots scores
def plotScore(scores):
    plt.figure()
    plt.plot(scores)
    plt.title("Score History")
    plt.xlabel("Episode Number")
    plt.show()


# Parameters
BATCH_SIZE = 128
LR = 1e-3
EPISODES = 5000
TARGET_SCORE = 250.
GAMMA = 0.99
MEMORY_SIZE = 10000
LEARN_STEP = 7
TAU = 2e-3
SAVE_NET = True

# Training
env = gym.make('LunarLander-v2')  # Make environment with LunarLander-v2
num_states = env.observation_space.shape[0]  # Get Number of state observations
print("num_states = " + str(num_states))
num_actions = env.action_space.n  # Number of actions possible
print("num_actions = " + str(num_actions))

agent = DQN(                # Initiate agent
    n_states=num_states,
    n_actions=num_actions,
    batch_size=BATCH_SIZE,
    lr=LR,
    gamma=GAMMA,
    mem_size=MEMORY_SIZE,
    learn_step=LEARN_STEP,
    tau=TAU,
)


# Train model

score_hist = []
epsilon_hist = []
max_steps = 500
eps_start = 1.0
eps_end = 0.1
eps_decay = 0.995
epsilon = eps_start

bar_format = '{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]'

pbar = trange(EPISODES, unit="ep", bar_format=bar_format, ascii=True)
for idx_epi in pbar:
    state = env.reset()
    score = 0
    for idx_step in range(max_steps):
        action = agent.getAction(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        agent.save2memory(state, action, reward, next_state, done)
        state = next_state
        score += reward

        if done:
            break

    score_hist.append(score)
    score_avg = np.mean(score_hist[-100:])  # Calculate average of last 100 scores
    epsilon_hist.append(epsilon)
    epsilon = max(eps_end, epsilon * eps_decay)  # Decay Epsilon
    pbar.set_postfix_str(f"Score: {score: 7.2f}, 100 score avg: {score_avg: 7.2f}, epsilon: {epsilon: 7.5f}")
    pbar.update(0)

    if len(score_hist) >= 100 and score_avg >= TARGET_SCORE:
        break

if (idx_epi + 1) < EPISODES:
    print("\nTarget Reached in " + str(idx_epi) + " episodes")
else:
    print("\nDone!")

if SAVE_NET:
    torch.save(agent.net_eval.state_dict(), 'checkpoint.pth')

# Plot Score History
plt.figure()
plt.plot(score_hist)
plt.title("Score History")
plt.xlabel("Episode Number")
plt.show()

# Plot Epsilon History
plt.figure()
plt.plot(epsilon_hist)
plt.title("Epsilon History")
plt.xlabel("Episode Number")
plt.show()

# Evaluate Final Model across 1000 runs
final_score_hist = []
for i in range(100):
    state = env.reset()
    this_score = 0
    for idx_step in range(500):
        action = agent.getAction(state, epsilon=0)
        state, reward, done, _ = env.step(action)
        this_score += reward
        if done:
            break
    final_score_hist.append(this_score)
env.close()
average_score = sum(final_score_hist) / len(final_score_hist)
print("Average score of model over " + str(len(final_score_hist)) + " runs = " + str(average_score))

## Used to save Gifs of trained model


def TextOnImg(img, score):
    img = Image.fromarray(img)
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), f"Score={score: .2f}", font=font, fill=(255, 255, 255))

    return np.array(img)


def save_frames_as_gif(frames, filename, path="gifs/"):
    if not os.path.exists(path):
        os.makedirs(path)

    print("Saving gif...", end="")
    imageio.mimsave(path + filename + ".gif", frames, fps=60)

    print("Done!")


def save_to_gif(frames, filename='./gifs/animation.gif'):

    print("Saving Gif...", end=" ")
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(filename, fps=60)
    print("Done")


def gym2gif(env, agent, filename="gym_animation", loop=3):
    frames = []
    for i in range(loop): # Loop through
        state = env.reset()
        score = 0
        for idx_step in range(500):
            frame = env.render(mode="rgb_array")
            frames.append(TextOnImg(frame, score))
            action = agent.getAction(state, epsilon=0)
            state, reward, done, _ = env.step(action)
            score += reward
            if done:
                break
    env.close()
    save_to_gif(frames, filename='./gifs/animation.gif')


gym2gif(env, agent, loop=3)