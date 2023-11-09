import os
import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import pytesseract


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='YOLOv8')
    parser.add_argument(
        '--webcam-resolution',
        default=[1920, 1080],
        nargs=2,
        type=int,
    )
    args = parser.parse_args()
    return args


def capture_webcam(output_file):
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

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
    cv2.destroyAllWindows()


def detect_dotnumbers():
    model = YOLO('data/NumberModel.pt')
    capture_webcam("img/test.png")
    img = cv2.imread("img/test.png")
    results = model(img, conf=0.6, save=True, save_crop=True)[0]
    detections = sv.Detections.from_ultralytics(results)
    # print(detections)

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    labels = [
        f"{model.names[class_id]} {confidence:.2f}"
        for _, _, confidence, class_id, _
        in detections
    ]

    img = box_annotator.annotate(
        scene=img,
        detections=detections,
        labels=labels,
    )

    cv2.imshow('yolov8', img)


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def noise_removal(image):
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = cv2.medianBlur(image, 3)
    return image


def thin_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image


def recognize_numbers():
    pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    images = []
    for filename in os.listdir("../runs/detect/predict7/crops/DotNumber"):
        if filename.endswith(".jpg"):
            images.append(filename)
    # print(images)

    for image in images:
        img = cv2.imread(f"../runs/detect/predict7/crops/DotNumber/{image}")
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

        # cv2.imshow("Result", img)
        # hImg, wImg, _ = img.shape
        config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789. -l train'
        boxes = pytesseract.image_to_data(img, config=config)
        for x, b in enumerate(boxes.splitlines()):
            if x != 0:
                b = b.split()
                print(b)
                if len(b) == 12:
                    x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                    cv2.rectangle(img, (x, y), (w + x, h + y), (0, 0, 255), 1)
                    cv2.putText(img, b[11], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
        cv2.imshow("Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # capture_webcam("img/test.png")
    # detect_dotnumbers()
    recognize_numbers()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
