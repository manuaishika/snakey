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

class SnakeGame:
    def __init__(self, width=10, height=10, block_size=30, render_mode='human'):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.render_mode = render_mode
        self.max_steps = width * height * 2  # Prevent infinite loops
        
        if render_mode == 'human':
            self.display = pygame.display.set_mode((width * block_size, height * block_size))
            pygame.display.set_caption('Snake DQN')
            self.clock = pygame.time.Clock()
        else:
            self.display = None
            self.clock = None
            
        self.reset()

    def reset(self):
        self.snake = [(self.width//2, self.height//2)]
        self.direction = (0, 1)
        self.food = self._place_food()
        self.score = 0
        self.game_over = False
        self.steps_since_food = 0
        return self._get_state()

    def _place_food(self):
        # More efficient food placement
        available_positions = []
        for x in range(self.width):
            for y in range(self.height):
                if (x, y) not in self.snake:
                    available_positions.append((x, y))
        return random.choice(available_positions) if available_positions else (0, 0)

    def _get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        # Enhanced state representation
        state = np.zeros(11, dtype=np.float32)
        
        # Normalized positions
        state[0] = head_x / self.width
        state[1] = head_y / self.height
        state[2] = food_x / self.width
        state[3] = food_y / self.height
        
        # Direction vector
        state[4] = self.direction[0]
        state[5] = self.direction[1]
        
        # Danger detection (straight, left, right relative to current direction)
        state[6] = int(self._is_danger_relative(0))    # straight
        state[7] = int(self._is_danger_relative(-1))   # left
        state[8] = int(self._is_danger_relative(1))    # right
        
        # Food direction relative to head
        state[9] = np.sign(food_x - head_x)  # -1, 0, 1
        state[10] = np.sign(food_y - head_y)  # -1, 0, 1
        
        return state

    def _is_danger_relative(self, turn):
        """Check danger relative to current direction: 0=straight, -1=left, 1=right"""
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left
        current_idx = directions.index(self.direction)
        new_idx = (current_idx + turn) % 4
        dx, dy = directions[new_idx]
        
        head_x, head_y = self.snake[0]
        next_x, next_y = head_x + dx, head_y + dy
        
        return (next_x < 0 or next_x >= self.width or
                next_y < 0 or next_y >= self.height or
                (next_x, next_y) in self.snake)

    def step(self, action):
        # Handle pygame events only if rendering
        if self.render_mode == 'human':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        # Actions: 0=straight, 1=right, 2=left
        if action == 1:  # Turn right
            directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            current_idx = directions.index(self.direction)
            self.direction = directions[(current_idx + 1) % 4]
        elif action == 2:  # Turn left
            directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            current_idx = directions.index(self.direction)
            self.direction = directions[(current_idx - 1) % 4]
        # action == 0 means go straight (no change)

        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        # Check collisions
        if (new_head[0] < 0 or new_head[0] >= self.width or
            new_head[1] < 0 or new_head[1] >= self.height or
            new_head in self.snake):
            self.game_over = True
            return self._get_state(), -10, True

        old_head = self.snake[0]
        self.snake.insert(0, new_head)
        reward = 0
        
        # Food collection
        if new_head == self.food:
            self.score += 1
            self.food = self._place_food()
            reward = 10
            self.steps_since_food = 0
        else:
            self.snake.pop()
            self.steps_since_food += 1

        # Improved reward system
        old_dist = abs(old_head[0] - self.food[0]) + abs(old_head[1] - self.food[1])  # Manhattan distance
        new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
        
        if new_dist < old_dist:
            reward += 1
        else:
            reward -= 1

        # Penalty for taking too long
        if self.steps_since_food > self.max_steps:
            reward -= 5
            self.game_over = True

        if self.render_mode == 'human':
            self._render()
            
        return self._get_state(), reward, self.game_over

    def _render(self):
        if self.display is None:
            return
            
        self.display.fill((0, 0, 0))
        for segment in self.snake:
            pygame.draw.rect(self.display, (0, 255, 0),
                            (segment[0] * self.block_size, segment[1] * self.block_size,
                             self.block_size, self.block_size))
        pygame.draw.rect(self.display, (255, 0, 0),
                         (self.food[0] * self.block_size, self.food[1] * self.block_size,
                          self.block_size, self.block_size))
        pygame.display.update()
        self.clock.tick(60)  # Increased FPS

# Improved DQN with batch normalization and dropout
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Prioritized Experience Replay (simplified)
class PrioritizedReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.priorities = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.memory[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = max_priority
            
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) == len(self.priorities):
            priorities = np.array(self.priorities)
            probabilities = priorities / priorities.sum()
            indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
            return [self.memory[i] for i in indices], indices
        else:
            indices = random.sample(range(len(self.memory)), batch_size)
            return [self.memory[i] for i in indices], indices

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)

# Enhanced DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Improved optimizer with learning rate scheduling
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)
        
        self.memory = PrioritizedReplayMemory(50000)  # Larger memory
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64  # Larger batch size
        self.target_update = 50  # More frequent updates
        self.steps = 0

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(3)  # 0, 1, 2 actions
            
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return
            
        transitions, indices = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))

        states = torch.FloatTensor(np.array(batch[0])).to(self.device)
        actions = torch.LongTensor(batch[1]).to(self.device)
        rewards = torch.FloatTensor(batch[2]).to(self.device)
        next_states = torch.FloatTensor(np.array(batch[3])).to(self.device)
        dones = torch.FloatTensor(batch[4]).to(self.device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN
        next_actions = self.model(next_states).argmax(1)
        next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Calculate TD errors for priority updates
        td_errors = torch.abs(current_q_values - target_q_values.detach())
        
        # Huber loss for stability
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()
        self.scheduler.step()

        # Update priorities
        self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

def main():
    # Train without rendering for speed
    env = SnakeGame(render_mode='none')
    agent = DQNAgent(state_dim=11, action_dim=3)
    episodes = 2000
    scores = []
    
    print(f"Training on {agent.device}")

    try:
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.memory.push(state, action, reward, next_state, done)
                
                # Train multiple times per step for better sample efficiency
                if len(agent.memory) >= agent.batch_size:
                    for _ in range(2):
                        agent.train()
                
                state = next_state
                total_reward += reward
                
            scores.append(env.score)
            
            # More frequent progress updates
            if episode % 50 == 0:
                avg_score = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
                print(f"Episode {episode}, Score: {env.score}, Avg Score: {avg_score:.2f}, "
                      f"Epsilon: {agent.epsilon:.3f}, LR: {agent.scheduler.get_last_lr()[0]:.5f}")
                
                # Save model periodically
                if episode % 200 == 0 and episode > 0:
                    torch.save(agent.model.state_dict(), f'snake_dqn_episode_{episode}.pth')
                    
    except KeyboardInterrupt:
        print("Training interrupted")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pygame.quit()

    print("Training completed!")
    
    # Save final model
    torch.save(agent.model.state_dict(), 'snake_dqn_final.pth')
    
    return scores

if __name__ == "__main__":
    main()
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
