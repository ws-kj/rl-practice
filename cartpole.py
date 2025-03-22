import gymnasium as gym
from gymnasium.wrappers import RecordVideo 
import torch
from torch import nn
from torch import optim 
import matplotlib.pyplot as plt

SIM_NAME = 'CartPole-v1'

default_hyperparams = {
    'learning_rate': 0.005,
    'discount_factor': 0.995,
    'exploration_prob': 0.0, # no epsilon greedy seems to work better
    'exploration_decay': 0.995,
    'epochs': 1000,
    'input_dim': 4, # pos, vel, theta, d_theta
    'output_dim': 2, # left, right
    'satisfaction_threshold': 50 # number of perfect runs to end training
}

# REINFORCE policy gradient
class PolicyNetwork(nn.Module):
    def __init__(self, observation_space, output_space):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(observation_space, 32),
            nn.ReLU(),
            nn.Linear(32, output_space),
            nn.Softmax(dim=-1)
        )
   
    def forward(self, x):
        return self.fc(x)

class Agent(object):
    def __init__(self, config=None):
        self.config = config if config else default_hyperparams
        self.policy = PolicyNetwork(
            observation_space=self.config['input_dim'], 
            output_space=self.config['output_dim']
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config['learning_rate'])
        self.epsilon = self.config['exploration_prob']

        self.loss_history, self.reward_history = [], [] # for visualization
        self.training = True

    def collect_trajectory(self, env):
        obs, info = env.reset()
        log_probs, rewards = [], []

        done = False
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32) # 1x4 
            action_probs = self.policy(obs_tensor)

            if torch.rand(1).item() < self.epsilon:
                action = env.action_space.sample() # eps-greedy sampling 
            else:
                action = torch.multinomial(action_probs, 1).item() # strict sample of action space
            log_probs.append(torch.log(action_probs[action]))
           
            self.epsilon *= self.config['exploration_decay']

            obs, reward, step_done, truncated, _ = env.step(action)
            done = step_done or (truncated and self.training) 
            rewards.append(reward)

        return log_probs, rewards

    def update_policy(self, log_probs, rewards):
        returns = []
        total_reward = 0

        for reward in reversed(rewards):
            total_reward = reward + total_reward  * self.config['discount_factor']
            returns.append(total_reward)
        returns.reverse() 


        returns = torch.tensor(returns, dtype=torch.float32)
        #returns = nn.functional.normalize(returns) # might not work?
        returns = (returns -returns.mean()) / (returns.std() + 1e-8) # normalize returns
        log_probs = torch.stack(log_probs)
        loss = torch.sum(-log_probs * (returns -returns.mean())) # baseline subtraction
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        self.loss_history.append(loss.item())
        self.reward_history.append(sum(rewards))

def plot_metrics(agent):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(agent.reward_history, label='Total Return')
    plt.xlabel('Episodes')
    plt.ylabel('Total Return')
    plt.title('Training Progress')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(agent.loss_history, label='Training Loss')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.title('Policy Loss Over Time')
    plt.legend()
    
    plt.show()

def run_sim(agent):
    rollout = False

    local_hist = 0
    env = gym.make(SIM_NAME)

    print(' -- Running ' + SIM_NAME + ' --\n') 
    for i in range(0, agent.config['epochs']):
        log_probs, rewards = agent.collect_trajectory(env)

        total_reward = sum(rewards)

        if not rollout:
            agent.update_policy(log_probs, rewards)
            if total_reward == 500:
                local_hist += 1
                # determine convergence if we hit n max reward episodes in a row
                if local_hist >= agent.config['satisfaction_threshold']: 
                    rollout = True 
                    env = gym.make(SIM_NAME, render_mode='rgb_array')
                    env = RecordVideo(env, video_folder="./videos")
                    agent.training = False
                    print("Convergence reached. Recording rollout...")
            else:
                local_hist = max(0, local_hist-1)
        else:
            print("Final return: " + str(total_reward))
            break

        print("Episode " + str(i) + " -- total reward: " + str(total_reward) + ' -- local: ' + str(local_hist))

    env.close()

if __name__ == "__main__":
    agent = Agent()
    run_sim(agent)
    plot_metrics(agent)
