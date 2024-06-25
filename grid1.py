import cv2
import sqlite3


# Function to store the grid representation in the database
def store_grid_representation(grid):
    # Connect to the database
    conn = sqlite3.connect("grid2.db")
    cursor = conn.cursor()

    # Create a table if not exists
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS GridRepresentation (
            id INTEGER PRIMARY KEY,
            grid_00 INTEGER,
            grid_01 INTEGER,
            grid_10 INTEGER,
            grid_11 INTEGER
        )
    """
    )

    # Insert the grid representation into the database
    cursor.execute(
        "INSERT INTO GridRepresentation (grid_00, grid_01, grid_10, grid_11) VALUES (?, ?, ?, ?)",
        (grid[0][0], grid[0][1], grid[1][0], grid[1][1]),
    )
    conn.commit()

    # Close the connection
    conn.close()


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

    # Initialize grid to store presence of points and set all values to zero
    grid = [[0] * GRID_COLS for _ in range(GRID_ROWS)]

    # Convert frame to grayscale for human detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect humans in the frame
    humans = human_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Draw circles at the center points of detected humans
    for x, y, w, h in humans:
        center_x = x + (w // 2)
        center_y = y + h
        center_coordinates = (center_x, center_y)
        radius = 3
        color = (255, 0, 0)  # Green color in BGR
        thickness = 2
        cv2.circle(frame, center_coordinates, radius, color, thickness)

    # Update grid based on detected humans
    for x, y, w, h in humans:
        center_x = x + (w // 2)
        center_y = y + h

        # Calculate grid cell index for the center point
        row_index = int(center_y / (frame.shape[0] / GRID_ROWS))
        col_index = int(center_x / (frame.shape[1] / GRID_COLS))

        # Ensure row_index and col_index are within grid bounds
        if 0 <= row_index < GRID_ROWS and 0 <= col_index < GRID_COLS:
            # Store the presence of human in the grid
            grid[row_index][col_index] = 1

    # Store the grid representation in the database
    store_grid_representation(grid)

    # Display the frame with detected humans
    cv2.imshow("Detected Humans", frame)

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

# Release video capture and close windows
video.release()
cv2.destroyAllWindows()
