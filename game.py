import random
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import math

# Initialize Pygame
pygame.init()

# Snake Game Environment
class SnakeGame:
    def __init__(self, width=10, height=10, block_size=30):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.display = pygame.display.set_mode((width * block_size, height * block_size))
        pygame.display.set_caption('Snake DQN')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.snake = [(5, 5)]
        self.direction = (0, 1)
        self.food = self._place_food()
        self.score = 0
        self.game_over = False
        return self._get_state()

    def _place_food(self):
        while True:
            food = (random.randint(0, self.width-1), random.randint(0, self.height-1))
            if food not in self.snake:
                return food

    def _get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        state = [
            head_x / self.width,
            head_y / self.height,
            food_x / self.width,
            food_y / self.height,
            int(self._is_danger(-1, 0)),
            int(self._is_danger(1, 0)),
            int(self._is_danger(0, -1)),
            int(self._is_danger(0, 1))
        ]
        return np.array(state, dtype=np.float32)

    def _is_danger(self, dx, dy):
        head_x, head_y = self.snake[0]
        next_x, next_y = head_x + dx, head_y + dy
        return (next_x < 0 or next_x >= self.width or
                next_y < 0 or next_y >= self.height or
                (next_x, next_y) in self.snake)

    def step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        if 0 <= action < 4:
            new_dir = directions[action]
            if new_dir[0] != -self.direction[0] or new_dir[1] != -self.direction[1]:
                self.direction = new_dir

        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        # Check for collision BEFORE modifying the snake
        if (new_head[0] < 0 or new_head[0] >= self.width or
            new_head[1] < 0 or new_head[1] >= self.height or
            new_head in self.snake):
            self.game_over = True
            return self._get_state(), -10, True

        # Store old head position for distance calculation
        old_head = self.snake[0]
        
        self.snake.insert(0, new_head)
        reward = 0
        
        if new_head == self.food:
            self.score += 1
            self.food = self._place_food()
            reward = 10
        else:
            self.snake.pop()

        # Distance-based reward calculation (fixed)
        new_dist = math.hypot(new_head[0] - self.food[0], new_head[1] - self.food[1])
        old_dist = math.hypot(old_head[0] - self.food[0], old_head[1] - self.food[1])
        
        if new_dist < old_dist:
            reward += 0.1
        elif new_dist > old_dist:
            reward -= 0.1

        self._render()
        return self._get_state(), reward, self.game_over

    def _render(self):
        self.display.fill((0, 0, 0))
        for segment in self.snake:
            pygame.draw.rect(self.display, (0, 255, 0),
                            (segment[0] * self.block_size, segment[1] * self.block_size,
                             self.block_size, self.block_size))
        pygame.draw.rect(self.display, (255, 0, 0),
                         (self.food[0] * self.block_size, self.food[1] * self.block_size,
                          self.block_size, self.block_size))
        pygame.display.update()
        self.clock.tick(10)

# DQN Model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = ReplayMemory(10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.target_update = 100
        self.steps = 0

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(4)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))

        states = torch.FloatTensor(batch[0]).to(self.device)
        actions = torch.LongTensor(batch[1]).to(self.device)
        rewards = torch.FloatTensor(batch[2]).to(self.device)
        next_states = torch.FloatTensor(batch[3]).to(self.device)
        dones = torch.FloatTensor(batch[4]).to(self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        targets = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

# Main Training Loop
def main():
    env = SnakeGame()
    agent = DQNAgent(state_dim=8, action_dim=4)
    episodes = 500
    scores = []

    try:
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.memory.push(state, action, reward, next_state, done)
                agent.train()
                state = next_state
                total_reward += reward
            scores.append(env.score)
            if episode % 10 == 0:
                print(f"Episode {episode}, Score: {env.score}, Avg Score: {np.mean(scores[-10:]):.2f}, Epsilon: {agent.epsilon:.2f}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pygame.quit()

    print("Training completed!")
    return scores

if __name__ == "__main__":
    main()
