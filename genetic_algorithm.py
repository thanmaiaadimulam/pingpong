import numpy as np
import random
from rl_agent import PongAgent
import tensorflow as tf

class GeneticAlgorithm:
    def __init__(self, population_size=50, state_size=8, action_size=3):
        self.population_size = population_size
        self.state_size = state_size
        self.action_size = action_size
        self.population = []
        self.generation = 0
        self.best_agent = None
        self.best_score = -float('inf')
        
        # Initialize population with more sophisticated agents
        for _ in range(population_size):
            agent = PongAgent(state_size, action_size)
            # Initialize with more sophisticated network
            agent.model = self._build_enhanced_model()
            self.population.append(agent)
    
    def _build_enhanced_model(self):
        # More sophisticated neural network with multiple layers and advanced features
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
                     optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0003),
                     run_eagerly=False)
        return model
    
    def evaluate_agent(self, agent1, agent2, games=10):
        """Make two agents play against each other and return their scores"""
        scores = []
        for _ in range(games):
            score1, score2 = self.play_game(agent1, agent2)
            scores.append((score1, score2))
        return np.mean(scores, axis=0)
    
    def play_game(self, agent1, agent2):
        """Simulate a game between two agents with enhanced physics and strategy"""
        # Initialize game state
        ball = type('Ball', (), {
            'x': 350, 'y': 250, 'x_vel': 5, 'y_vel': 0,
            'reset': lambda: None
        })
        paddle1 = type('Paddle', (), {'x': 10, 'y': 200, 'height': 100})
        paddle2 = type('Paddle', (), {'x': 680, 'y': 200, 'height': 100})
        
        score1, score2 = 0, 0
        max_steps = 3000  # Increased max steps for longer games
        
        # Strategy tracking
        last_paddle1_y = paddle1.y
        last_paddle2_y = paddle2.y
        paddle1_strategy = []
        paddle2_strategy = []
        
        for _ in range(max_steps):
            # Get states with strategy information
            state1 = get_state(ball, paddle1, paddle1_strategy)
            state2 = get_state(ball, paddle2, paddle2_strategy)
            
            # Get actions with strategy consideration
            action1 = agent1.act(state1)
            action2 = agent2.act(state2)
            
            # Enhanced paddle movement with momentum
            paddle_speed = 4
            if action1 == 0 and paddle1.y > 0:  # Up
                paddle1.y -= paddle_speed
            elif action1 == 1 and paddle1.y < 400:  # Down
                paddle1.y += paddle_speed
                
            if action2 == 0 and paddle2.y > 0:  # Up
                paddle2.y -= paddle_speed
            elif action2 == 1 and paddle2.y < 400:  # Down
                paddle2.y += paddle_speed
            
            # Track strategy
            paddle1_strategy.append(paddle1.y - last_paddle1_y)
            paddle2_strategy.append(paddle2.y - last_paddle2_y)
            if len(paddle1_strategy) > 10:
                paddle1_strategy.pop(0)
                paddle2_strategy.pop(0)
            last_paddle1_y = paddle1.y
            last_paddle2_y = paddle2.y
            
            # Enhanced ball physics with spin
            ball.x += ball.x_vel
            ball.y += ball.y_vel
            
            # Ball collision with top and bottom with enhanced physics
            if ball.y <= 0 or ball.y >= 500:
                ball.y_vel *= -1
                # Add spin effect
                if abs(ball.y_vel) < 3:
                    ball.y_vel += random.uniform(-1, 1)
            
            # Enhanced paddle collision with spin and angle calculation
            if (ball.x <= 30 and abs(ball.y - paddle1.y) < 50) or \
               (ball.x >= 670 and abs(ball.y - paddle2.y) < 50):
                ball.x_vel *= -1
                # Calculate reflection angle based on where ball hits paddle
                paddle_center = paddle1.y + 50 if ball.x <= 30 else paddle2.y + 50
                relative_intersect_y = (paddle_center - ball.y) / 50
                reflection_angle = relative_intersect_y * 0.7  # Increased maximum reflection angle
                
                # Add spin based on paddle movement
                paddle_vel = paddle1_strategy[-1] if ball.x <= 30 else paddle2_strategy[-1]
                spin_effect = paddle_vel * 0.1
                
                ball.y_vel = -reflection_angle * 5 + spin_effect
                ball.x_vel += random.uniform(-0.3, 0.3)  # Reduced randomness for more predictable bounces
            
            # Scoring
            if ball.x <= 0:
                score2 += 1
                ball.reset()
            elif ball.x >= 700:
                score1 += 1
                ball.reset()
            
            if score1 >= 10 or score2 >= 10:
                break
        
        return score1, score2
    
    def crossover(self, parent1, parent2):
        """Enhanced crossover with mutation"""
        child = PongAgent(self.state_size, self.action_size)
        child.model = self._build_enhanced_model()
        
        # Get weights from both parents
        weights1 = parent1.model.get_weights()
        weights2 = parent2.model.get_weights()
        
        # Create new weights with enhanced crossover
        child_weights = []
        for w1, w2 in zip(weights1, weights2):
            # Use weighted average instead of random selection
            alpha = np.random.random(w1.shape)
            child_weights.append(alpha * w1 + (1 - alpha) * w2)
            
            # Enhanced mutation
            if random.random() < 0.2:  # Increased mutation rate
                mutation = np.random.normal(0, 0.2, w1.shape)  # Increased mutation strength
                child_weights[-1] += mutation
        
        child.model.set_weights(child_weights)
        return child
    
    def evolve(self, generations=50):
        """Enhanced evolution process"""
        for gen in range(generations):
            print(f"\nGeneration {gen + 1}")
            
            # Enhanced evaluation
            scores = []
            for i in range(self.population_size):
                agent_scores = []
                for j in range(self.population_size):
                    if i != j:
                        score1, score2 = self.evaluate_agent(self.population[i], self.population[j], games=10)  # More games per evaluation
                        agent_scores.append(score1 - score2)
                avg_score = np.mean(agent_scores)
                scores.append((avg_score, i))
            
            # Sort by performance
            scores.sort(reverse=True)
            
            # Keep track of best agent
            if scores[0][0] > self.best_score:
                self.best_score = scores[0][0]
                self.best_agent = self.population[scores[0][1]]
                print(f"New best score: {self.best_score}")
                # Save best agent more frequently
                self.best_agent.save(f"best_agent_gen_{gen}.h5")
            
            # Create new population with enhanced selection
            new_population = []
            
            # Keep top 30% of agents (increased from 20%)
            elite_size = max(3, self.population_size // 3)
            for i in range(elite_size):
                new_population.append(self.population[scores[i][1]])
            
            # Fill rest of population with children of top performers
            while len(new_population) < self.population_size:
                # Tournament selection
                tournament_size = 5
                tournament = random.sample(new_population[:elite_size], tournament_size)
                parent1 = max(tournament, key=lambda x: x.model.get_weights()[0].mean())
                tournament = random.sample(new_population[:elite_size], tournament_size)
                parent2 = max(tournament, key=lambda x: x.model.get_weights()[0].mean())
                
                child = self.crossover(parent1, parent2)
                new_population.append(child)
            
            self.population = new_population
            self.generation += 1
        
        return self.best_agent

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