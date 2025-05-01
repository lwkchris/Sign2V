import cv2

cap = cv2.VideoCapture(0)  # Default camera index

if not cap.isOpened():
    print("Error: Cannot access the camera.")
    cap.release()
    exit()

print("Camera is accessible. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from the camera.")
        break

    cv2.imshow("Camera Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()