// Game constants
const WIDTH = 1280;
const HEIGHT = 720;
const PADDLE_WIDTH = 20;
const PADDLE_HEIGHT = 150;
const BALL_RADIUS = 15;
const BALL_SPEED = 7;
const AI_SPEED = 15;
const AI_REACTION_TIME = 0.1;

// Game state
let ballX = WIDTH / 2;
let ballY = HEIGHT / 2;
let ballVelX = BALL_SPEED;
let ballVelY = BALL_SPEED / 2;
let leftPaddleY = HEIGHT / 2 - PADDLE_HEIGHT / 2;
let rightPaddleY = HEIGHT / 2 - PADDLE_HEIGHT / 2;
let leftScore = 0;
let rightScore = 0;
let gameRunning = false;
let aiReactionTimer = 0;
let lastTime = 0;

// Canvas setup
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
canvas.width = WIDTH;
canvas.height = HEIGHT;

// MediaPipe Hands setup
const hands = new Hands({
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
    }
});

hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7
});

// Camera setup
const video = document.getElementById('cameraFeed');
const startButton = document.getElementById('startButton');

async function setupCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                resolve(video);
            };
        });
    } catch (err) {
        console.error("Error accessing camera:", err);
        alert("Could not access camera. Please make sure you have a camera connected and permissions are granted.");
    }
}

// Game functions
function resetBall(direction) {
    ballX = WIDTH / 2;
    ballY = HEIGHT / 2;
    
    if (direction === "left") {
        ballVelX = -BALL_SPEED;
    } else if (direction === "right") {
        ballVelX = BALL_SPEED;
    } else {
        ballVelX = Math.random() > 0.5 ? BALL_SPEED : -BALL_SPEED;
    }
    
    ballVelY = (Math.random() - 0.5) * BALL_SPEED;
}

function aiControlPaddle(dt) {
    if (ballVelX < 0) {
        aiReactionTimer += dt;
        if (aiReactionTimer >= AI_REACTION_TIME) {
            aiReactionTimer = 0;
            const targetY = ballY - PADDLE_HEIGHT / 2;
            const distance = targetY - leftPaddleY;
            
            if (Math.abs(distance) > 0) {
                const move = Math.min(AI_SPEED * dt * 60, Math.abs(distance)) * (distance > 0 ? 1 : -1);
                leftPaddleY += move;
            }
        }
    }
}

function draw() {
    // Clear canvas
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, WIDTH, HEIGHT);
    
    // Draw center line
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(WIDTH / 2, 0);
    ctx.lineTo(WIDTH / 2, HEIGHT);
    ctx.stroke();
    
    // Draw paddles
    ctx.fillStyle = 'rgba(0, 255, 0, 0.7)'; // AI paddle (green)
    ctx.fillRect(100 - PADDLE_WIDTH / 2, leftPaddleY, PADDLE_WIDTH, PADDLE_HEIGHT);
    
    ctx.fillStyle = 'rgba(255, 0, 0, 0.7)'; // Human paddle (red)
    ctx.fillRect(WIDTH - (100 - PADDLE_WIDTH / 2), rightPaddleY, PADDLE_WIDTH, PADDLE_HEIGHT);
    
    // Draw ball
    ctx.fillStyle = 'rgba(0, 0, 255, 0.7)';
    ctx.beginPath();
    ctx.arc(ballX, ballY, BALL_RADIUS, 0, Math.PI * 2);
    ctx.fill();
    
    // Draw scores
    ctx.fillStyle = 'white';
    ctx.font = '74px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(leftScore.toString(), WIDTH / 4, 100);
    ctx.fillText(rightScore.toString(), 3 * WIDTH / 4, 100);
    
    // Draw labels
    ctx.font = '36px Arial';
    ctx.fillText('AI', 100, 50);
    ctx.fillText('HUMAN', WIDTH - 100, 50);
}

function update(dt) {
    // Ball movement
    ballX += ballVelX * dt * 60;
    ballY += ballVelY * dt * 60;
    
    // Ball collision with top and bottom
    if (ballY <= BALL_RADIUS) {
        ballY = BALL_RADIUS;
        ballVelY = Math.abs(ballVelY);
    } else if (ballY >= HEIGHT - BALL_RADIUS) {
        ballY = HEIGHT - BALL_RADIUS;
        ballVelY = -Math.abs(ballVelY);
    }
    
    // Scoring
    if (ballX >= WIDTH - BALL_RADIUS) {
        leftScore++;
        resetBall("left");
    } else if (ballX <= BALL_RADIUS) {
        rightScore++;
        resetBall("right");
    }
    
    // Paddle boundaries
    leftPaddleY = Math.max(0, Math.min(HEIGHT - PADDLE_HEIGHT, leftPaddleY));
    rightPaddleY = Math.max(0, Math.min(HEIGHT - PADDLE_HEIGHT, rightPaddleY));
    
    // Paddle collisions
    // Left paddle
    if (ballX - BALL_RADIUS <= 100 + PADDLE_WIDTH / 2 && ballVelX < 0) {
        if (leftPaddleY <= ballY && ballY <= leftPaddleY + PADDLE_HEIGHT) {
            ballX = 100 + PADDLE_WIDTH / 2 + BALL_RADIUS;
            ballVelX = Math.abs(ballVelX) * 1.05;
            
            const relativeIntersectY = (leftPaddleY + PADDLE_HEIGHT / 2) - ballY;
            const normalizedRelativeIntersectY = relativeIntersectY / (PADDLE_HEIGHT / 2);
            ballVelY = -normalizedRelativeIntersectY * BALL_SPEED;
        }
    }
    
    // Right paddle
    if (ballX + BALL_RADIUS >= WIDTH - (100 + PADDLE_WIDTH / 2) && ballVelX > 0) {
        if (rightPaddleY <= ballY && ballY <= rightPaddleY + PADDLE_HEIGHT) {
            ballX = WIDTH - (100 + PADDLE_WIDTH / 2) - BALL_RADIUS;
            ballVelX = -Math.abs(ballVelX) * 1.05;
            
            const relativeIntersectY = (rightPaddleY + PADDLE_HEIGHT / 2) - ballY;
            const normalizedRelativeIntersectY = relativeIntersectY / (PADDLE_HEIGHT / 2);
            ballVelY = -normalizedRelativeIntersectY * BALL_SPEED;
        }
    }
    
    // Cap ball speed
    const maxSpeed = 15;
    if (Math.abs(ballVelX) > maxSpeed) {
        ballVelX = maxSpeed * (ballVelX > 0 ? 1 : -1);
    }
    
    // AI control
    aiControlPaddle(dt);
}

function gameLoop(timestamp) {
    if (!gameRunning) return;
    
    const dt = (timestamp - lastTime) / 1000;
    lastTime = timestamp;
    
    update(dt);
    draw();
    
    requestAnimationFrame(gameLoop);
}

// Hand tracking
hands.onResults((results) => {
    if (!gameRunning) return;
    
    if (results.multiHandLandmarks) {
        for (let i = 0; i < results.multiHandLandmarks.length; i++) {
            const handedness = results.multiHandedness[i].classification[0].label;
            const landmarks = results.multiHandLandmarks[i];
            
            if (handedness === 'Right') {
                const yPosition = landmarks[8].y * HEIGHT;
                rightPaddleY = yPosition - PADDLE_HEIGHT / 2;
            }
        }
    }
});

// Start game
startButton.addEventListener('click', async () => {
    startButton.style.display = 'none';
    gameRunning = true;
    lastTime = performance.now();
    await setupCamera();
    hands.send({ image: video });
    gameLoop(performance.now());
    
    // Continuously send video frames to MediaPipe
    setInterval(() => {
        if (gameRunning) {
            hands.send({ image: video });
        }
    }, 100);
});

// Handle window resize
window.addEventListener('resize', () => {
    const scale = Math.min(
        window.innerWidth / WIDTH,
        window.innerHeight / HEIGHT
    );
    canvas.style.transform = `scale(${scale})`;
}); 