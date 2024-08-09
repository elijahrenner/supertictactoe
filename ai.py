import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os

# Define the environment for Super Tic Tac Toe
class SuperTicTacToe:
    def __init__(self):
        self.large_board = self.initialize_boards()
        self.current_player = 1
        self.last_move = (1, 1)
        self.done = False

    def initialize_boards(self):
        small_board = np.zeros((3, 3), dtype=int)
        large_board = np.array([[small_board.copy() for _ in range(3)] for _ in range(3)])
        return large_board

    def reset(self):
        self.large_board = self.initialize_boards()
        self.current_player = 1
        self.last_move = (1, 1)
        self.done = False
        return self.get_state()

    def get_state(self):
        return self.large_board.flatten()

    def make_move(self, large_row, large_col, small_row, small_col):
        if 0 <= large_row < 3 and 0 <= large_col < 3 and 0 <= small_row < 3 and 0 <= small_col < 3:
            if self.large_board[large_row, large_col][small_row, small_col] == 0:
                self.large_board[large_row, large_col][small_row, small_col] = self.current_player
                self.last_move = (small_row, small_col)
                if self.check_large_board_win():
                    self.done = True
                    return self.get_state(), 1 if self.current_player == 1 else -1, self.done
                self.current_player = 3 - self.current_player
                return self.get_state(), -0.01, self.done
            else:
                return self.get_state(), -1, True  # Invalid move
        else:
            return self.get_state(), -1, True  # Invalid move

    def check_small_board_win(self, small_board):
        for i in range(3):
            if np.all(small_board[i, :] == small_board[i, 0]) and small_board[i, 0] != 0:
                return small_board[i, 0]
            if np.all(small_board[:, i] == small_board[0, i]) and small_board[0, i] != 0:
                return small_board[0, i]
        if small_board[0, 0] == small_board[1, 1] == small_board[2, 2] and small_board[0, 0] != 0:
            return small_board[0, 0]
        if small_board[0, 2] == small_board[1, 1] == small_board[2, 0] and small_board[0, 2] != 0:
            return small_board[0, 2]
        return 0

    def check_large_board_win(self):
        large_status = np.zeros((3, 3), dtype=int)
        for i in range(3):
            for j in range(3):
                large_status[i, j] = self.check_small_board_win(self.large_board[i, j])
        
        return self.check_small_board_win(large_status) != 0

    def render(self):
        def print_board(board):
            symbols = {0: ".", 1: "X", 2: "O"}
            for row in board:
                print(" ".join(symbols[cell] for cell in row))
        
        for row in range(3):
            for sub_row in range(3):
                for col in range(3):
                    print(" ".join(str(self.large_board[row, col][sub_row][i]) for i in range(3)), end=" | ")
                print()
            print("-" * 13)

# Define the DQN model
# Define the DQN model with increased learning rate and larger batch size
class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005  # Increased learning rate
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='relu'))  # Larger network
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, env):
        valid_move = False
        while not valid_move:
            if np.random.rand() <= self.epsilon:
                action = random.randrange(self.action_size)
            else:
                act_values = self.model.predict(state)
                action = np.argmax(act_values[0])

            small_row = (action // 9) % 3
            small_col = action % 3
            large_row = env.last_move[0]
            large_col = env.last_move[1]

            # Check if the chosen move is valid
            if env.large_board[large_row, large_col][small_row, small_col] == 0:
                valid_move = True
            else:
                print(f"AI attempted invalid move at ({small_row}, {small_col}) in mini-grid ({large_row}, {large_col}). Re-selecting...")

        return action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name, save_format='keras')

# Function for human player to make a move
def human_move(env):
    large_row, large_col = env.last_move
    print(f"Player must play in mini-grid ({large_row}, {large_col})")
    while True:
        try:
            small_row = int(input("Enter small row (0-2): "))
            small_col = int(input("Enter small col (0-2): "))
            if 0 <= small_row < 3 and 0 <= small_col < 3:
                state, reward, done = env.make_move(large_row, large_col, small_row, small_col)
                if reward == -1:
                    print("Invalid move, try again.")
                else:
                    return state, reward, done
            else:
                print("Invalid input, try again.")
        except ValueError:
            print("Invalid input, try again.")

# Train the model
def train_model(agent, env, episodes=1000, batch_size=64, patience=100):
    best_reward = -float('inf')
    patience_counter = 0

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])
        total_reward = 0

        for time in range(500):
            action = agent.act(state, env)
            large_row = env.last_move[0]
            large_col = env.last_move[1]
            small_row = action // 9
            small_col = action % 9
            next_state, reward, done = env.make_move(large_row, large_col, small_row, small_col)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, agent.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        # Check if we have a new best reward
        if total_reward > best_reward:
            best_reward = total_reward
            patience_counter = 0
            print(f"New best reward: {best_reward} at episode {e+1}")
        else:
            patience_counter += 1

        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}, Patience: {patience_counter}/{patience}")

        # If we've exceeded patience, stop training
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

        # Decay epsilon after each episode
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

# Play the game against the trained AI
# Play the game against the trained AI
# Correct action mapping and move validation
def play_against_ai(agent, env):
    state = env.reset()
    state = np.reshape(state, [1, agent.state_size])
    while not env.done:
        env.render()  # Display the board
        if env.current_player == 1:
            print("AI's turn.")
            action = agent.act(state, env)
            small_row = (action // 9) % 3
            small_col = action % 3
            large_row = env.last_move[0]
            large_col = env.last_move[1]
            print(f"AI chooses position ({small_row}, {small_col}) in mini-grid ({large_row}, {large_col}).")

            next_state, reward, done = env.make_move(large_row, large_col, small_row, small_col)
            print(f"Reward received: {reward}, Done: {done}")
            next_state = np.reshape(next_state, [1, agent.state_size])
            state = next_state
            if done:
                env.render()
                if reward == 1:
                    print("AI wins!")
                else:
                    print("It's a draw!")
                break
        else:
            state, reward, done = human_move(env)
            state = np.reshape(state, [1, agent.state_size])
            if done:
                env.render()
                if reward == 1:
                    print("You win!")
                else:
                    print("It's a draw!")
                break

if __name__ == "__main__":
    env = SuperTicTacToe()
    state_size = 81
    action_size = 81
    agent = DQN(state_size, action_size)

    # Load the trained model if you want to continue training from an existing model
    # agent.load("supertictactoe_dqn.keras")

    # Train the model with early stopping
    train_model(agent, env, episodes=1000, batch_size=64, patience=10)

    # Save the trained model
    agent.save("supertictactoe_dqn_2.keras")

    # Play against the AI
    play_against_ai(agent, env)
    env = SuperTicTacToe()
    state_size = 81
    action_size = 81
    agent = DQN(state_size, action_size)

    # Load the trained model
    agent.load("supertictactoe_dqn.keras")

    # Play against the AI
    play_against_ai(agent, env)