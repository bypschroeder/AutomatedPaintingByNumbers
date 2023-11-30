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
    # capture_webcam("img/test.png")
    img = cv2.imread("img/test.png")
    results = model(img, conf=0.6, save=True, save_crop=True)[0]
    detections = sv.Detections.from_ultralytics(results)

    images = []
    # TODO: path has to be renamed to current predict folder
    for filename in os.listdir("../runs/detect/predict/crops/DotNumber"):
        if filename.endswith(".jpg"):
            images.append(filename)
    images.sort(key=lambda x: int(x.split('image')[1].split('.jpg')[0]))

    coordinates = []
    for bbox in detections.xyxy:
        x_min, y_min, x_max, y_max = bbox.tolist()

        coordinates.append((int(x_min), int(y_min), int(x_max), int(y_max)))

    image_coordinates = []
    for coord in coordinates:
        if images:
            image_filename = images.pop(0)
            image_coordinates.append((image_filename, coord))

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
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(image_coordinates)
    return image_coordinates


def convert_coordinates(coordinates):
    converted_coordinates = []

    for img_path, number, bbox, dot_coordinates in coordinates:
        new_x_min = None
        new_y_min = None
        new_x_max = None
        new_y_max = None
        if dot_coordinates is not None:
            dot_x_min = bbox[0] + dot_coordinates[0]
            dot_y_min = bbox[1] + dot_coordinates[1]
            dot_x_max = bbox[0] + dot_coordinates[2]
            dot_y_max = bbox[1] + dot_coordinates[3]
            if dot_x_min & dot_x_max > 420:
                dot_x_min -= 420
                dot_x_max -= 420
            new_x_min = round((dot_x_min / 1080) * 100, 2)
            new_y_min = round((dot_y_min / 1080) * 100, 2)
            new_x_max = round((dot_x_max / 1080) * 100, 2)
            new_y_max = round((dot_y_max / 1080) * 100, 2)

        converted_coordinates.append((img_path, number, (new_x_min, new_y_min, new_x_max, new_y_max)))

    print(converted_coordinates)
    return converted_coordinates


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


def recognize_numbers(img_coordinates):
    pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    detection_info = []
    test = []

    for img_info, coordinates in img_coordinates:
        img = cv2.imread(f"../runs/detect/predict/crops/DotNumber/{img_info}")
        # img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

        # cv2.imshow("Result", img)
        # hImg, wImg, _ = img.shape
        config_number = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789 -l train'
        detected_number = pytesseract.image_to_string(img, config=config_number)

        cleaned_number = ''.join(filter(str.isdigit, detected_number))

        try:
            number = int(cleaned_number)
        except ValueError:
            number = None

        config_dot = r'--oem 3 --psm 6 -c tessedit_char_whitelist=., -l train'
        detected_dot = pytesseract.image_to_data(img, config=config_dot)
        dot_coordinates = None

        for x, b in enumerate(detected_dot.splitlines()):
            if x != 0:
                b = b.split()
                # print(b)
                if len(b) == 12:
                    confidence = float(b[10]) if b[10] != '-1' else 0
                    if confidence > 50:
                        x_min, y_min, x_max, y_max = int(b[6]), int(b[7]), int(b[6]) + int(b[8]), int(b[7]) + int(b[9])
                        dot_coordinates = (x_min, y_min, x_max, y_max)
                        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
                        cv2.putText(img, b[11], (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)

        # print(boxes)
        detection_info.append((img_info, number, coordinates, dot_coordinates))
        print(detection_info)
        test = convert_coordinates(detection_info)
        print(test)
        cv2.imshow("Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    sorted_data = sorted(test, key=lambda item: item[1])

    print(sorted_data)
    return sorted_data


if __name__ == "__main__":
    # capture_webcam("img/test.png")
    # detect_dotnumbers()
    data = recognize_numbers(detect_dotnumbers())

    img_path, number, coordinates = data[4]
    print(coordinates)

    img = cv2.imread("img/test.png")
    dot = [int((coordinates[0] / 100) * 1080) + 420, int((coordinates[1] / 100) * 1080), int((coordinates[2] / 100) * 1080) + 420, int((coordinates[3] / 100) * 1080)]
    # cv2.rectangle(img, (dot[0], dot[1]), (dot[2], dot[3]), (0, 0, 255), 1)1
    # middle_x = int((dot[0] + dot[2]) / 2)
    # middle_y = int((dot[1] + dot[3]) / 2)
    # cv2.circle(img, (middle_x, middle_y), 1, (0, 0, 255), 1)
    cv2.circle(img, (dot[0], dot[1]), 1, (0, 0, 255), 2)
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # convert_coordinates(detect_dotnumbers())
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
