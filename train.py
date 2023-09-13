import gym
import torch
import pickle
import cv2
import imageio
import random
import torch.optim as optim
import numpy as np
import torch.nn as nn
from model import IMLNN
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

def train(epoch, name, env, expert, policy, optimizer, loss_fn, num_epochs, device, render_interval, writer, batch_size):
    policy.train()
    epoch_loss = 0
    epoch_reward = 0
    obs, _ = env.reset()

    if epoch % render_interval == 0 or epoch == 499:
        if epoch == 499:
            e = 500
        else:
            e = epoch
        frames = []
        done = False
        action_step = 0
        while not done:
            action = policy(torch.FloatTensor(obs).to(device)).detach().cpu().numpy()
            obs, _, terminated, truncated, _ = env.step(action)
            
            frame = env.render()
            frames.append(frame)
            imageio.mimsave(f'simulations/{name}_simulation_epoch_{e}.gif', frames)

            action_step += 1
            if action_step > 100:
                done = True

            if done:
                print(f'video recorded for epoch {e}')
                break

    num_batches = len(expert) // batch_size

    for batch_idx in range(num_batches):
        batch_states, batch_actions = zip(*random.sample(expert, batch_size))
        batch_states = torch.FloatTensor(batch_states).to(device)
        batch_actions = torch.FloatTensor(batch_actions).to(device)

        predicted_actions = policy(batch_states)


        batch_rewards = np.zeros(batch_size)

        for i in range(batch_size):
            _, reward, _, _, _ = env.step(predicted_actions[i].detach().cpu().numpy())
            batch_rewards[i] = reward

        loss = loss_fn(batch_actions, predicted_actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_reward += sum(batch_rewards)
        
    print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss / num_batches}')
    print(f'Epoch [{epoch + 1}/{num_epochs}] - Reward: {epoch_reward}')
    writer.add_scalar('training loss', epoch_loss / num_batches)
    writer.add_scalar('reward', epoch_reward)
        
    env.close()
    writer.close()


def test(name, env, policy):
    sim_states = []
    sim_actions = []

    for test in range(3):
        obs, _ = env.reset()
        done = False
        policy.eval()

        frames = []
        action_step = 0
        while not done:
            action = policy(torch.FloatTensor(obs).to(device)).detach().cpu().numpy()
            obs, _, terminated, truncated, _ = env.step(action)

            sim_states.append(obs)
            sim_actions.append(action)

            frame = env.render()

            frames.append(frame)
            imageio.mimsave(f'simulations/{name}_simulation_test_{test}.gif', frames)

            action_step += 1
            if action_step > 100:
                done = True

            if done:
                break

        env.close()
    
    return sim_states, sim_actions

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    expert_file = 'expert_data_Ant-v4.pkl'
    environment = 'Ant-v4'
    num_epochs = 500
    learning_rate = 1e-3
    render_interval = 100
    batch_size = 32
    writer = SummaryWriter('runs/imitation-learning')
    name = 'train'
    model_name = 'Ant'
    env = gym.make(environment, render_mode = 'rgb_array')

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    policy = IMLNN(input_size, output_size).to(device)

    with open(expert_file, 'rb') as f:
        expert1, expert2 = pickle.load(f)

    assert len(expert1['observation']) == len(expert1['action'])
    assert len(expert2['observation']) == len(expert2['action'])

    expert_states = []
    expert_actions = []
    for num in range(len(expert1['observation'])):
        X = expert1['observation'][num][:27]
        Y = expert2['observation'][num][:27]
        expert_states.append(X)
        expert_states.append(Y)
        expert_actions.append(expert1['action'][num])
        expert_actions.append(expert2['action'][num])
    
    assert len(expert_states) == len(expert_actions)

    expert = list(zip(expert_states, expert_actions))
    random.shuffle(expert)

    for epoch in range(num_epochs):
        optimizer = optim.Adam(policy.parameters(), lr = learning_rate)
        loss_fn = nn.MSELoss()

        if epoch % 50 == 0 and epoch != 0:
            learning_rate = 0.9 * learning_rate

        train(epoch, name, env, expert, policy, optimizer, loss_fn, num_epochs, device, render_interval, writer, batch_size)
                
    print('trainig is done')

    torch.save({'model_state_dict': policy.state_dict()}, f'checkpoints/{model_name}_policy_checkpoint.pth')
    name = 'test'
    sim_states, sim_actions = test(name, env, policy)

    #####################################################################################################################################################

    expert_states = np.array(expert_states)
    sim_states = np.array(sim_states)
    DAgger_states = np.concatenate((expert_states, sim_states), axis = 0)
    DAgger_actions = np.concatenate((expert_actions, sim_actions), axis = 0)

    DAgger_expert = list(zip(DAgger_states, DAgger_actions))
    random.shuffle(DAgger_expert)

    optimizer = optim.Adam(policy.parameters(), lr = learning_rate)

    name = 'DTrain'
    epoch = 0
    learning_rate = 1e-3

    for epoch in range(num_epochs):
        optimizer = optim.Adam(policy.parameters(), lr = learning_rate)
        loss_fn = nn.MSELoss()

        if epoch % 50 == 0 and epoch != 0:
            learning_rate = 0.9 * learning_rate

        train(epoch, name, env, expert, policy, optimizer, loss_fn, num_epochs, device, render_interval, writer, batch_size)
    
    print('trainig is done')

    torch.save({'model_state_dict': policy.state_dict()}, f'checkpoints/{model_name}_DAgger_policy_checkpoint.pth')
    name = 'DTest'
    sim_states, sim_actions = test(name, env, policy)

    writer.close()
    print('simulation is done!')