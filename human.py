import cv2

# Load pre-trained Haar cascade for human detection
human_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_fullbody.xml"
)

# Load the input video
video = cv2.VideoCapture("contentvideo.mp4")

# Define grid parameters
GRID_ROWS = 2
GRID_COLS = 2

# Loop through the video frames
while True:
    # Read a frame from the video
    ret, frame = video.read()
    if not ret:
        break  # If no frame is returned, break the loop

    # Convert frame to grayscale for human detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initialize grid to store presence of humans
    grid = [[0] * GRID_COLS for _ in range(GRID_ROWS)]

    # Detect humans in the frame
    humans = human_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Update grid based on detected humans
    for x, y, w, h in humans:
        # Calculate grid cell index
        row_index = int((y + h / 2) / (frame.shape[0] / GRID_ROWS))
        col_index = int((x + w / 2) / (frame.shape[1] / GRID_COLS))
        # Mark grid cell as containing human
        grid[row_index][col_index] = 1

    # Display the grid
    print("Grid Representation:")
    for row in grid:
        print(row)

    # Display the frame with detected humans
    cv2.imshow("Detected Humans", frame)

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

# Release video capture and close windows
video.release()
cv2.destroyAllWindows()
