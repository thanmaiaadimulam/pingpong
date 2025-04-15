import pygame
import numpy as np
from rl_agent import PongAgent, get_state
from genetic_algorithm import GeneticAlgorithm
from neat_agent import NEATAgent, run_neat, load_best_agent
import os

pygame.init()


WIDTH, HEIGHT = 700, 500
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong")

FPS = 60

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

PADDLE_WIDTH, PADDLE_HEIGHT = 20, 100
BALL_RADIUS = 7

SCORE_FONT = pygame.font.SysFont("comicsans", 50)
WINNING_SCORE = 10


class Paddle:
    COLOR = WHITE
    VEL = 4

    def __init__(self, x, y, width, height):
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.width = width
        self.height = height

    def draw(self, win):
        pygame.draw.rect(
            win, self.COLOR, (self.x, self.y, self.width, self.height))

    def move(self, up=True):
        if up:
            self.y -= self.VEL
        else:
            self.y += self.VEL

    def reset(self):
        self.x = self.original_x
        self.y = self.original_y


class Ball:
    MAX_VEL = 5
    COLOR = WHITE

    def __init__(self, x, y, radius):
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.radius = radius
        self.x_vel = self.MAX_VEL
        self.y_vel = 0

    def draw(self, win):
        pygame.draw.circle(win, self.COLOR, (self.x, self.y), self.radius)

    def move(self):
        self.x += self.x_vel
        self.y += self.y_vel

    def reset(self):
        self.x = self.original_x
        self.y = self.original_y
        self.y_vel = 0
        self.x_vel *= -1


def draw(win, paddles, ball, left_score, right_score):
    win.fill(BLACK)

    left_score_text = SCORE_FONT.render(f"{left_score}", 1, WHITE)
    right_score_text = SCORE_FONT.render(f"{right_score}", 1, WHITE)
    win.blit(left_score_text, (WIDTH//4 - left_score_text.get_width()//2, 20))
    win.blit(right_score_text, (WIDTH * (3/4) -
                                right_score_text.get_width()//2, 20))

    for paddle in paddles:
        paddle.draw(win)

    for i in range(10, HEIGHT, HEIGHT//20):
        if i % 2 == 1:
            continue
        pygame.draw.rect(win, WHITE, (WIDTH//2 - 5, i, 10, HEIGHT//20))

    ball.draw(win)
    pygame.display.update()


def handle_collision(ball, left_paddle, right_paddle):
    if ball.y + ball.radius >= HEIGHT:
        ball.y_vel *= -1
    elif ball.y - ball.radius <= 0:
        ball.y_vel *= -1

    if ball.x_vel < 0:
        if ball.y >= left_paddle.y and ball.y <= left_paddle.y + left_paddle.height:
            if ball.x - ball.radius <= left_paddle.x + left_paddle.width:
                ball.x_vel *= -1

                middle_y = left_paddle.y + left_paddle.height / 2
                difference_in_y = middle_y - ball.y
                reduction_factor = (left_paddle.height / 2) / ball.MAX_VEL
                y_vel = difference_in_y / reduction_factor
                ball.y_vel = -1 * y_vel

    else:
        if ball.y >= right_paddle.y and ball.y <= right_paddle.y + right_paddle.height:
            if ball.x + ball.radius >= right_paddle.x:
                ball.x_vel *= -1

                middle_y = right_paddle.y + right_paddle.height / 2
                difference_in_y = middle_y - ball.y
                reduction_factor = (right_paddle.height / 2) / ball.MAX_VEL
                y_vel = difference_in_y / reduction_factor
                ball.y_vel = -1 * y_vel


def handle_paddle_movement(keys, left_paddle, right_paddle, left_agent=None, right_agent=None, ball=None):
    if keys[pygame.K_w] and left_paddle.y - left_paddle.VEL >= 0:
        left_paddle.move(up=True)
    if keys[pygame.K_s] and left_paddle.y + left_paddle.VEL + left_paddle.height <= HEIGHT:
        left_paddle.move(up=False)

    # AI controlled right paddle
    if right_agent and ball:
        if isinstance(right_agent, NEATAgent):
            state = right_agent.get_state(ball, right_paddle)
        else:
            state = get_state(ball, right_paddle)
        action = right_agent.act(state)
        
        if action == 0 and right_paddle.y - right_paddle.VEL >= 0:  # Move up
            right_paddle.move(up=True)
        elif action == 1 and right_paddle.y + right_paddle.VEL + right_paddle.height <= HEIGHT:  # Move down
            right_paddle.move(up=False)

    # AI controlled left paddle
    if left_agent and ball:
        if isinstance(left_agent, NEATAgent):
            state = left_agent.get_state(ball, left_paddle)
        else:
            state = get_state(ball, left_paddle)
        action = left_agent.act(state)
        
        if action == 0 and left_paddle.y - left_paddle.VEL >= 0:  # Move up
            left_paddle.move(up=True)
        elif action == 1 and left_paddle.y + left_paddle.VEL + left_paddle.height <= HEIGHT:  # Move down
            left_paddle.move(up=False)


def main():
    run = True
    clock = pygame.time.Clock()

    # Initialize game mode
    print("Select game mode:")
    print("1. Train NEAT AI")
    print("2. Play against trained NEAT AI")
    print("3. Watch NEAT AI vs NEAT AI")
    print("4. Train RL AI")
    print("5. Play against trained RL AI")
    print("6. Watch RL AI vs RL AI")
    mode = input("Enter mode (1-6): ")

    # Initialize agents
    left_agent = None
    right_agent = None

    if mode in ["1", "2", "3"]:
        # NEAT AI modes
        if mode == "1":
            # Training mode
            print("Training NEAT AI...")
            run_neat("neat_config.txt")
            return
        else:
            try:
                right_agent = load_best_agent("neat_config.txt")
                if mode == "3":
                    left_agent = load_best_agent("neat_config.txt")
                print("Loaded trained NEAT AI")
            except Exception as e:
                print(f"Error loading NEAT AI: {e}")
                print("No trained NEAT AI found. Please train first using mode 1.")
                return
    else:
        # RL AI modes
        if mode == "4":
            # Training mode
            ga = GeneticAlgorithm(population_size=25)
            best_agent = ga.evolve(generations=25)
            print("Training complete! Best agent saved as 'best_agent.h5'")
            best_agent.save("best_agent.h5")
            return
        else:
            try:
                right_agent = PongAgent(8, 3)
                right_agent.load("best_agent.h5")
                if mode == "6":
                    left_agent = PongAgent(8, 3)
                    left_agent.load("best_agent.h5")
                print("Loaded trained RL AI")
            except Exception as e:
                print(f"Error loading RL AI: {e}")
                print("No trained RL AI found. Please train first using mode 4.")
                return

    left_paddle = Paddle(10, HEIGHT//2 - PADDLE_HEIGHT//2, PADDLE_WIDTH, PADDLE_HEIGHT)
    right_paddle = Paddle(WIDTH - 10 - PADDLE_WIDTH, HEIGHT//2 - PADDLE_HEIGHT//2, PADDLE_WIDTH, PADDLE_HEIGHT)
    ball = Ball(WIDTH // 2, HEIGHT // 2, BALL_RADIUS)

    left_score = 0
    right_score = 0

    while run:
        clock.tick(FPS)
        draw(WIN, [left_paddle, right_paddle], ball, left_score, right_score)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

        keys = pygame.key.get_pressed()
        handle_paddle_movement(keys, left_paddle, right_paddle, left_agent, right_agent, ball)

        ball.move()
        handle_collision(ball, left_paddle, right_paddle)

        if ball.x < 0:
            right_score += 1
            ball.reset()
        elif ball.x > WIDTH:
            left_score += 1
            ball.reset()

        if left_score >= WINNING_SCORE or right_score >= WINNING_SCORE:
            text = "Left Player Won!" if left_score >= WINNING_SCORE else "Right Player Won!"
            draw(WIN, [left_paddle, right_paddle], ball, left_score, right_score)
            pygame.display.update()
            pygame.time.delay(5000)
            ball.reset()
            left_paddle.reset()
            right_paddle.reset()
            left_score = 0
            right_score = 0

    pygame.quit()


if __name__ == '__main__':
    main()