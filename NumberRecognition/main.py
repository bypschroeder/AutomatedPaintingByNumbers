import os
import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import pytesseract
import math
import shutil


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
    folder_path = '../runs/detect/'

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        print(f"Error: {folder_path} does not exist")

    # TODO: Improve model accuracy by adding more images to dataset
    model = YOLO('data/NumberModel2.pt')
    # capture_webcam("img/test.png")
    img = cv2.imread('img/4.jpg')
    results = model(img, conf=0.3, save=True, save_crop=True)[0]
    detections = sv.Detections.from_ultralytics(results)

    numbers = []
    for filename in os.listdir("../runs/detect/predict/crops/Number"):
        if filename.endswith(".jpg"):
            numbers.append(filename)
    numbers.sort(key=lambda x: int(x.split('image')[1].split('.jpg')[0]))

    dots = []
    for filename in os.listdir("../runs/detect/predict/crops/Dot"):
        if filename.endswith(".jpg"):
            dots.append(filename)
    dots.sort(key=lambda x: int(x.split('image')[1].split('.jpg')[0]))

    classes = []
    for _, _, _, class_id, _ in detections:
        classes.append(class_id)

    coords = []
    for bbox in detections.xyxy:
        x_min, y_min, x_max, y_max = bbox
        coords.append((x_min, y_min, x_max, y_max))

    number_coords = []
    dot_coords = []

    number_index = 0
    dot_index = 0

    for i in range(len(classes)):
        if classes[i] == 1:
            number_coords.append(('number', coords[i], numbers[number_index]))
            number_index += 1
        elif classes[i] == 0:
            dot_coords.append(('dot', coords[i], dots[dot_index]))
            dot_index += 1

    def calculate_distance(coord1, coord2):
        x1, y1, _, _ = coord1[1]
        x2, y2, _, _ = coord2[1]
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    grouped_coords = []

    for number_coord in number_coords:
        for dot_coord in dot_coords:
            if calculate_distance(number_coord, dot_coord) < 100:
                grouped_coords.append((number_coord, dot_coord))

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

    return grouped_coords


def convert_coordinates(data):
    converted_data = []

    for number_coords, dot_coords, number in data:
        _, (dot_x_min, dot_y_min, dot_x_max, dot_y_max), _ = dot_coords

        center_x = (dot_x_min + dot_x_max) / 2
        center_y = (dot_y_min + dot_y_max) / 2

        if center_x > 420:
            center_x -= 420
        new_center_x = round((center_x / 1080) * 100, 2)
        new_center_y = round((center_y / 1080) * 100, 2)

        converted_data.append((number, (new_center_x, new_center_y)))

    return converted_data


def recognize_numbers(img_coordinates):
    pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    detection_info = []
    converted_coordinates = []

    for number_info, dot_info in img_coordinates:
        number_img = cv2.imread(f"../runs/detect/predict/crops/Number/{number_info[2]}")
        number_img = cv2.resize(number_img, (0, 0), fx=2, fy=2)
        number_img = cv2.cvtColor(number_img, cv2.COLOR_BGR2GRAY)
        number_img = cv2.threshold(number_img, 125, 255, cv2.THRESH_BINARY)[1]
        config_number = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
        detected_number = pytesseract.image_to_string(number_img, config=config_number)
        cleaned_number = ''.join(filter(str.isdigit, detected_number))

        try:
            number = int(cleaned_number)
        except ValueError:
            number = None

        detection_info.append((number_info, dot_info, number))
        print(number)
        converted_coordinates = convert_coordinates(detection_info)
        cv2.imshow("Result", number_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    sorted_data = sorted(converted_coordinates, key=lambda item: item[0])

    return sorted_data


if __name__ == "__main__":
    # capture_webcam("img/test.png")

    # detect_dotnumbers()

    data = recognize_numbers(detect_dotnumbers())
    print(data)
    img = cv2.imread("img/4.jpg")

    for number, coordinates in data:
        x, y = coordinates
        x = int((x / 100) * 1080) + 420
        y = int((y / 100) * 1080)

        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), 3)

    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # convert_coordinates(detect_dotnumbers())
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
