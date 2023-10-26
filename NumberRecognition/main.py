import cv2


def capture_webcam(output_file):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    ret, frame = cap.read()

    if ret:
        cv2.imwrite(output_file, frame)
        print(f"Image saved as {output_file}")
    else:
        print("Failed to capture image")

    cap.release()


if __name__ == "__main__":
    capture_webcam("img/test.png")
