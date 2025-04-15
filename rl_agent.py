import numpy as np
import tensorflow as tf
from collections import deque
import random

class PongAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # Increased memory size
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0003  # Reduced learning rate
        self.training_frequency = 4
        self.frame_count = 0
        self.last_action = 1  # Start with stay
        self.strategy_history = deque(maxlen=10)  # Track recent movements
        self.model = self._build_model()

    def _build_model(self):
        # Enhanced neural network with multiple layers
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(48, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', 
                     optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate),
                     run_eagerly=False)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Action persistence with frame skipping and strategy consideration
        self.frame_count += 1
        
        # Update strategy history
        if hasattr(self, 'last_y'):
            self.strategy_history.append(state[0][7] - self.last_y)  # Track vertical movement
            if len(self.strategy_history) > 10:
                self.strategy_history.popleft()
        self.last_y = state[0][7]
        
        if self.frame_count % 3 != 0:  # Keep the same action for 3 frames
            return self.last_action
            
        if np.random.rand() <= self.epsilon:
            # Add some strategy to random actions
            if self.strategy_history:
                avg_strategy = np.mean(self.strategy_history)
                if abs(avg_strategy) > 0.1:  # If there's a clear strategy
                    self.last_action = 0 if avg_strategy < 0 else 1
                else:
                    self.last_action = random.randrange(self.action_size)
            else:
                self.last_action = random.randrange(self.action_size)
            return self.last_action
            
        act_values = self.model.predict(state, verbose=0)
        self.last_action = np.argmax(act_values[0])
        return self.last_action

    def replay(self, batch_size):
        self.frame_count += 1
        if self.frame_count % self.training_frequency != 0:
            return
            
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        # Batch processing for better performance
        states = np.array([x[0][0] for x in minibatch])
        next_states = np.array([x[3][0] for x in minibatch])
        
        # Predict all at once
        targets = self.model.predict(states, verbose=0)
        next_targets = self.model.predict(next_states, verbose=0)
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(next_targets[i])
            targets[i][action] = target
            
        # Single fit call with larger batch size
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def get_state(ball, paddle, strategy_history=None):
    # Enhanced state space with strategy information
    state = [
        (ball.x - paddle.x) / 700,  # Relative x distance
        (ball.y - paddle.y) / 500,  # Relative y distance
        ball.x_vel / 5,  # x velocity
        ball.y_vel / 5,  # y velocity
        paddle.y / 500,  # Paddle position
        (ball.y - (paddle.y + paddle.height/2)) / 500,  # Relative vertical position to paddle center
        ball.x_vel * ball.y_vel / 25,  # Combined velocity effect
        (ball.y - 250) / 250,  # Ball's vertical position relative to center
    ]
    
    # Add strategy information
    if strategy_history:
        avg_strategy = np.mean(strategy_history) if strategy_history else 0
        state.append(avg_strategy / 4)  # Normalized strategy
    else:
        state.append(0)
    
    return np.reshape(state, [1, 9]) 