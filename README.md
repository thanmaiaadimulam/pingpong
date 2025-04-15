# PingPongu - AI vs Human

A web-based Pong game where you play against an AI using hand tracking! Control the paddle with your right hand and try to beat the AI opponent.

## How to Download and Run Locally

1. **Download the Game**:
   - Click the green "Code" button at the top of this repository
   - Select "Download ZIP"
   - Extract the ZIP file to a folder on your computer

2. **Run the Game**:
   - Open the extracted folder
   - Double-click on `index.html` to open it in your default web browser
   - Alternatively, you can use a local server:
     ```bash
     # Using Python (if installed)
     python -m http.server 8000
     # Then open http://localhost:8000 in your browser
     ```

3. **Play the Game**:
   - Open the game in a modern browser (Chrome recommended)
   - Allow camera access when prompted
   - Click the "Start Game" button
   - Use your right hand to control the red paddle
   - The green paddle is controlled by the AI

## Online Play

You can also play the game online at: [https://thanmaiaadimulam.github.io/pingpong/](https://thanmaiaadimulam.github.io/pingpong/)

## Features

- Hand tracking using MediaPipe
- AI opponent with adjustable difficulty
- Real-time camera feed
- Score tracking
- Responsive design

## Technical Requirements

- Modern web browser (Chrome, Firefox, Edge)
- Webcam
- Good lighting conditions for hand tracking
- Internet connection (for hand tracking functionality)

## Troubleshooting

If you encounter any issues:
1. Make sure you're using a modern browser (Chrome recommended)
2. Ensure your webcam is properly connected and working
3. Grant camera permissions when prompted
4. Check that you have good lighting for hand tracking
5. If the game doesn't start, try refreshing the page

## Development

This project uses:
- HTML5 Canvas for rendering
- MediaPipe for hand tracking
- JavaScript for game logic

## License

MIT License 