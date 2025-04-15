import cv2
import mediapipe as mp
import pygame
import random
import time
import math

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)  # Set height

# MediaPipe hand tracking setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Pygame initialization
pygame.init()
WIDTH, HEIGHT = 1280, 720
wn = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('PingPongu - AI vs Human')
clock = pygame.time.Clock()  # For controlling frame rate
FPS = 60

# Colors
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

# Ball settings
radius = 15
ball_x, ball_y = WIDTH / 2, HEIGHT / 2
ball_speed = 7  # Base speed
ball_vel_x, ball_vel_y = ball_speed, ball_speed / 2

# Paddle dimensions
paddle_width, paddle_height = 20, 150
left_paddle_y = right_paddle_y = HEIGHT / 2 - paddle_height / 2
left_paddle_x, right_paddle_x = 100 - paddle_width / 2, WIDTH - (100 - paddle_width / 2)

# Score
left_score = 0
right_score = 0
font = pygame.font.SysFont(None, 74)
small_font = pygame.font.SysFont(None, 36)

# Smoothing for paddle movement (using a simple moving average)
right_paddle_positions = [HEIGHT / 2] * 5  # Last 5 positions

# AI settings
AI_SPEED = 15  # Maximum speed of AI paddle (increased from 8)
AI_REACTION_TIME = 0.1  # Delay in seconds before AI reacts
ai_reaction_timer = 0

# For calculating delta time between frames
last_time = time.time()


def reset_ball(direction=None):
    """Reset the ball to the center and set its velocity"""
    global ball_x, ball_y, ball_vel_x, ball_vel_y

    ball_x, ball_y = WIDTH / 2, HEIGHT / 2

    # Randomize the starting direction a bit
    if direction == "left":
        ball_vel_x = -ball_speed
    elif direction == "right":
        ball_vel_x = ball_speed
    else:
        ball_vel_x = ball_speed if random.random() > 0.5 else -ball_speed

    # Randomize the y velocity slightly
    ball_vel_y = random.uniform(-ball_speed / 2, ball_speed / 2)


def update_paddle_position(positions, new_position):
    """Update the position array and return the smoothed position"""
    positions.pop(0)  # Remove oldest position
    positions.append(new_position)  # Add new position
    return sum(positions) / len(positions)  # Return average


def draw_scores():
    """Draw the scores on the screen"""
    left_text = font.render(str(left_score), True, WHITE)
    right_text = font.render(str(right_score), True, WHITE)
    wn.blit(left_text, (WIDTH / 4, 50))
    wn.blit(right_text, (3 * WIDTH / 4, 50))


def ai_control_paddle(dt):
    """Control the left paddle using AI"""
    global left_paddle_y, ai_reaction_timer
    
    # Only update AI position if ball is moving towards AI paddle
    if ball_vel_x < 0:
        # Add reaction time delay
        ai_reaction_timer += dt
        if ai_reaction_timer >= AI_REACTION_TIME:
            ai_reaction_timer = 0
            
            # Calculate target position (center of paddle should align with ball)
            target_y = ball_y - paddle_height / 2
            
            # Calculate distance to move
            distance = target_y - left_paddle_y
            
            # Move paddle with speed limit
            if abs(distance) > 0:
                move = min(AI_SPEED * dt * 60, abs(distance)) * (1 if distance > 0 else -1)
                left_paddle_y += move


def draw_player_labels():
    """Draw labels showing which side is AI and which is human"""
    ai_text = small_font.render("AI", True, GREEN)
    human_text = small_font.render("HUMAN", True, WHITE)
    wn.blit(ai_text, (50, 20))
    wn.blit(human_text, (WIDTH - 150, 20))


# Main game loop
run = True
while run:
    # Calculate delta time for smooth movement regardless of frame rate
    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time

    # Process video from webcam
    success, img = cap.read()
    if not success:
        continue  # Skip this iteration if we couldn't get a frame

    img = cv2.flip(img, 1)  # Mirror image for more intuitive control
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Draw hand landmarks on the camera feed
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get handedness (Left/Right)
            handedness = results.multi_handedness[hand_idx].classification[0].label

            # Draw landmarks on the image
            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS,
                                  mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                                  mpDraw.DrawingSpec(color=(255, 0, 255), thickness=2))

            # Use index finger (landmark 8) to control paddle
            y_position = hand_landmarks.landmark[8].y * HEIGHT

            if handedness == 'Right':
                right_paddle_y = update_paddle_position(right_paddle_positions, y_position)

    # Handle pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    # Convert camera feed to pygame surface to use as background
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert back to RGB for pygame
    img = cv2.resize(img, (WIDTH, HEIGHT))  # Ensure it matches the game window size

    # Convert OpenCV image to Pygame surface
    camera_surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))

    # Display camera feed as background
    wn.blit(camera_surface, (0, 0))

    # Ball movement, adjusted for delta time
    ball_x += ball_vel_x * dt * 60  # Scale by 60 to maintain similar speed at 60 FPS
    ball_y += ball_vel_y * dt * 60

    # Ball collision with top and bottom walls
    if ball_y <= radius:
        ball_y = radius  # Prevent ball from going off-screen
        ball_vel_y = abs(ball_vel_y)  # Bounce downward
    elif ball_y >= HEIGHT - radius:
        ball_y = HEIGHT - radius  # Prevent ball from going off-screen
        ball_vel_y = -abs(ball_vel_y)  # Bounce upward

    # Ball goes past paddles - scoring
    if ball_x >= WIDTH - radius:
        left_score += 1
        reset_ball("left")
    elif ball_x <= radius:
        right_score += 1
        reset_ball("right")

    # Paddle boundary checks
    right_paddle_y = max(0, min(HEIGHT - paddle_height, right_paddle_y))
    left_paddle_y = max(0, min(HEIGHT - paddle_height, left_paddle_y))

    # Control AI paddle
    ai_control_paddle(dt)

    # Collision with paddles
    # Left paddle collision
    if ball_x - radius <= left_paddle_x + paddle_width and ball_vel_x < 0:
        if left_paddle_y <= ball_y <= left_paddle_y + paddle_height:
            ball_x = left_paddle_x + paddle_width + radius  # Prevent sticking
            ball_vel_x = abs(ball_vel_x) * 1.05  # Speed up slightly on each hit

            # Change angle based on where ball hits paddle
            relative_intersect_y = (left_paddle_y + (paddle_height / 2)) - ball_y
            normalized_relative_intersect_y = relative_intersect_y / (paddle_height / 2)
            ball_vel_y = -normalized_relative_intersect_y * ball_speed

    # Right paddle collision
    if ball_x + radius >= right_paddle_x and ball_vel_x > 0:
        if right_paddle_y <= ball_y <= right_paddle_y + paddle_height:
            ball_x = right_paddle_x - radius  # Prevent sticking
            ball_vel_x = -abs(ball_vel_x) * 1.05  # Speed up slightly and reverse

            # Change angle based on where ball hits paddle
            relative_intersect_y = (right_paddle_y + (paddle_height / 2)) - ball_y
            normalized_relative_intersect_y = relative_intersect_y / (paddle_height / 2)
            ball_vel_y = -normalized_relative_intersect_y * ball_speed

    # Cap the maximum ball speed
    max_speed = 15
    if abs(ball_vel_x) > max_speed:
        ball_vel_x = max_speed * (1 if ball_vel_x > 0 else -1)

    # Drawing with semi-transparency for better visibility over camera feed
    # Create a transparent surface for game elements
    game_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    # Draw game elements on the transparent surface
    pygame.draw.circle(game_surface, (BLUE[0], BLUE[1], BLUE[2], 220), (int(ball_x), int(ball_y)), radius=radius)
    pygame.draw.rect(game_surface, (GREEN[0], GREEN[1], GREEN[2], 180),
                     pygame.Rect(left_paddle_x, left_paddle_y, paddle_width, paddle_height))
    pygame.draw.rect(game_surface, (RED[0], RED[1], RED[2], 180),
                     pygame.Rect(right_paddle_x, right_paddle_y, paddle_width, paddle_height))

    # Draw center line with transparency
    pygame.draw.line(game_surface, (WHITE[0], WHITE[1], WHITE[2], 150), (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), 2)

    # Overlay game elements on the camera feed
    wn.blit(game_surface, (0, 0))

    # Draw scores and player labels
    draw_scores()
    draw_player_labels()

    # Update display
    pygame.display.update()

    # Control frame rate
    clock.tick(FPS)

# Clean up
cap.release()
pygame.quit()