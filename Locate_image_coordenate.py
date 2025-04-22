import cv2

# Path to your image
imagen_path = "Image Useful\\pose_landmarks_index.png"  # <-- change this to your image

# Load image
imagen = cv2.imread(imagen_path)

# Function called on mouse click
def Show_coordenadas(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordinates: x={x}, y={y}")
        # Draw a circle at the click position
        cv2.circle(imagen, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(imagen, f"({x},{y})", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow("Interactive Image", imagen)

# Show the image and start listening for mouse events
cv2.imshow("Interactive Image", imagen)
cv2.setMouseCallback("Interactive Image", Show_coordenadas)

# Wait until a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
