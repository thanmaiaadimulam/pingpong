import numpy as np
import neat
import os
import pickle
import random
from collections import deque
import pygame

class NEATAgent:
    def __init__(self, genome, config):
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.fitness = 0
        self.strategy_history = deque(maxlen=30)  # Increased history
        self.last_y = None
        self.frame_count = 0
        self.last_action = 1
        self.ball_trajectory = deque(maxlen=10)  # Increased trajectory history
        self.successful_hits = 0
        self.consecutive_hits = 0
        self.max_consecutive_hits = 0
        self.misses = 0
        self.total_balls = 0
        self.hit_ratio = 0.0
        self.avg_distance_to_ball = 0.0
        self.total_distance = 0.0
        self.steps = 0
        self.velocity_history = deque(maxlen=10)
        self.last_positions = deque(maxlen=5)

    def act(self, state):
        self.frame_count += 1
        self.steps += 1
        
        # Update histories
        if self.last_y is not None:
            self.strategy_history.append(state[0][7] - self.last_y)
            self.ball_trajectory.append((state[0][0], state[0][1]))
            if len(self.ball_trajectory) > 1:
                dx = self.ball_trajectory[-1][0] - self.ball_trajectory[-2][0]
                dy = self.ball_trajectory[-1][1] - self.ball_trajectory[-2][1]
                self.velocity_history.append((dx, dy))
        self.last_y = state[0][7]
        
        if self.frame_count % 2 != 0:
            return self.last_action
            
        # Enhanced state representation
        strategy_state = list(state[0])
        
        # Add strategy information
        if self.strategy_history:
            avg_strategy = np.mean(self.strategy_history)
            recent_strategy = np.mean(list(self.strategy_history)[-5:])
            strategy_state.extend([avg_strategy / 4, recent_strategy / 4])
        else:
            strategy_state.extend([0, 0])
            
        # Add ball trajectory information
        if len(self.ball_trajectory) > 1:
            trajectory = np.array(self.ball_trajectory)
            dx = trajectory[-1][0] - trajectory[0][0]
            dy = trajectory[-1][1] - trajectory[0][1]
            strategy_state.extend([dx, dy])
            
            # Add velocity trends
            if len(self.velocity_history) > 1:
                vel_array = np.array(self.velocity_history)
                avg_vel_x = np.mean(vel_array[:, 0])
                avg_vel_y = np.mean(vel_array[:, 1])
                strategy_state.extend([avg_vel_x, avg_vel_y])
            else:
                strategy_state.extend([0, 0])
        else:
            strategy_state.extend([0, 0, 0, 0])
        
        # Ensure we have exactly 16 inputs
        while len(strategy_state) < 16:
            strategy_state.append(0)
        strategy_state = strategy_state[:16]
        
        # Get action from NEAT network with additional processing
        output = self.net.activate(strategy_state)
        
        # Process network output for smoother control
        if max(output) - min(output) < 0.2:  # If outputs are close, maintain current action
            action = self.last_action
        else:
            action = np.argmax(output)
            
            # Add momentum to movements
            if action != self.last_action and len(self.last_positions) > 2:
                # Check if we're already moving in a good direction
                current_trend = np.mean(np.diff(list(self.last_positions)))
                if (current_trend < 0 and action == 0) or (current_trend > 0 and action == 1):
                    action = self.last_action
        
        self.last_action = action
        return action

    def get_state(self, ball, paddle):
        # Enhanced state representation
        state = [
            (ball.x - paddle.x) / 700,  # Relative x distance
            (ball.y - paddle.y) / 500,  # Relative y distance
            ball.x_vel / 5,  # x velocity
            ball.y_vel / 5,  # y velocity
            paddle.y / 500,  # Paddle position
            (ball.y - (paddle.y + paddle.height/2)) / 500,  # Relative vertical position to paddle center
            ball.x_vel * ball.y_vel / 25,  # Combined velocity effect
            (ball.y - 250) / 250,  # Ball's vertical position relative to center
            ball.x_vel / abs(ball.x_vel) if ball.x_vel != 0 else 0,  # Ball direction
            (paddle.y + paddle.height/2 - ball.y) / 500,  # Distance to paddle center
            min(abs(ball.y - paddle.y), abs(ball.y - (paddle.y + paddle.height))) / 500,  # Distance to paddle edges
            ball.x / 700,  # Ball's x position
            ball.y / 500,  # Ball's y position
            (paddle.y + paddle.height/2) / 500,  # Paddle center position
            self.consecutive_hits / 20,  # Normalized consecutive hits
            self.hit_ratio if hasattr(self, 'hit_ratio') else 0  # Current hit ratio
        ]
        return np.reshape(state, [1, 16])

def run_neat(config_file):
    """Run NEAT training with both paddles controlled by AI"""
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)
    
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    def eval_genomes(genomes, config):
        """Evaluate all genomes by having them play against each other"""
        # Create a list of agents from genomes
        agents = []
        for genome_id, genome in genomes:
            genome.fitness = 0
            agents.append((genome, NEATAgent(genome, config)))
        
        # Each agent plays against a random subset of other agents
        for i, (genome1, agent1) in enumerate(agents):
            # Play against 5 random opponents
            opponents = random.sample([a for j, a in enumerate(agents) if j != i], min(5, len(agents)-1))
            
            for genome2, agent2 in opponents:
                pygame.init()
                WIDTH, HEIGHT = 700, 500
                screen = pygame.Surface((WIDTH, HEIGHT))
                
                # Initialize game objects
                left_paddle = Paddle(10, HEIGHT//2 - 50, 20, 100)
                right_paddle = Paddle(WIDTH - 30, HEIGHT//2 - 50, 20, 100)
                ball = Ball(WIDTH//2, HEIGHT//2, 7)
                
                # Game loop
                max_steps = 2000
                steps = 0
                left_score = 0
                right_score = 0
                
                while steps < max_steps:
                    steps += 1
                    
                    # Get actions from both agents
                    left_state = agent1.get_state(ball, left_paddle)
                    right_state = agent2.get_state(ball, right_paddle)
                    
                    left_action = agent1.act(left_state)
                    right_action = agent2.act(right_state)
                    
                    # Update paddle positions
                    if left_action == 0 and left_paddle.y > 0:
                        left_paddle.y -= 4
                    elif left_action == 1 and left_paddle.y < HEIGHT - 100:
                        left_paddle.y += 4
                        
                    if right_action == 0 and right_paddle.y > 0:
                        right_paddle.y -= 4
                    elif right_action == 1 and right_paddle.y < HEIGHT - 100:
                        right_paddle.y += 4
                    
                    # Update ball
                    ball.x += ball.x_vel
                    ball.y += ball.y_vel
                    
                    # Ball collision with top and bottom
                    if ball.y + ball.radius >= HEIGHT or ball.y - ball.radius <= 0:
                        ball.y_vel *= -1
                    
                    # Ball collision with paddles
                    # Left paddle
                    if (ball.x - ball.radius <= left_paddle.x + 20 and
                        ball.y >= left_paddle.y and ball.y <= left_paddle.y + 100):
                        ball.x = left_paddle.x + 20 + ball.radius
                        ball.x_vel *= -1
                        
                        middle_y = left_paddle.y + 50
                        difference_in_y = middle_y - ball.y
                        reduction_factor = 50 / ball.MAX_VEL
                        ball.y_vel = -1 * difference_in_y / reduction_factor
                        
                        genome1.fitness += 10
                        agent1.successful_hits += 1
                        agent1.consecutive_hits += 1
                    
                    # Right paddle
                    if (ball.x + ball.radius >= right_paddle.x and
                        ball.y >= right_paddle.y and ball.y <= right_paddle.y + 100):
                        ball.x = right_paddle.x - ball.radius
                        ball.x_vel *= -1
                        
                        middle_y = right_paddle.y + 50
                        difference_in_y = middle_y - ball.y
                        reduction_factor = 50 / ball.MAX_VEL
                        ball.y_vel = -1 * difference_in_y / reduction_factor
                        
                        genome2.fitness += 10
                        agent2.successful_hits += 1
                        agent2.consecutive_hits += 1
                    
                    # Ball goes past paddles
                    if ball.x < 0:
                        right_score += 1
                        genome2.fitness += 5
                        ball.reset()
                        agent1.consecutive_hits = 0
                    elif ball.x > WIDTH:
                        left_score += 1
                        genome1.fitness += 5
                        ball.reset()
                        agent2.consecutive_hits = 0
                    
                    # Positioning reward
                    if ball.x_vel > 0:  # Ball moving right
                        if abs(right_paddle.y + 50 - ball.y) < 30:
                            genome2.fitness += 0.1
                    else:  # Ball moving left
                        if abs(left_paddle.y + 50 - ball.y) < 30:
                            genome1.fitness += 0.1
                    
                    # Early termination if one side is dominating
                    if abs(left_score - right_score) > 5:
                        break
                
                pygame.quit()
                
                # Final fitness adjustments
                if agent1.successful_hits > 0:
                    genome1.fitness += (agent1.successful_hits * 5) + (agent1.consecutive_hits * 2)
                if agent2.successful_hits > 0:
                    genome2.fitness += (agent2.successful_hits * 5) + (agent2.consecutive_hits * 2)
    
    winner = p.run(eval_genomes, 50)
    
    # Save the winner
    with open('best_neat_agent.pkl', 'wb') as f:
        pickle.dump(winner, f)
    
    return winner

class Ball:
    MAX_VEL = 5
    
    def __init__(self, x, y, radius):
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.radius = radius
        self.x_vel = self.MAX_VEL
        self.y_vel = 0
    
    def reset(self):
        self.x = self.original_x
        self.y = self.original_y
        self.y_vel = 0
        self.x_vel *= -1

class Paddle:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

def load_best_agent(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)
    
    with open('best_neat_agent.pkl', 'rb') as f:
        winner = pickle.load(f)
    
    return NEATAgent(winner, config) 