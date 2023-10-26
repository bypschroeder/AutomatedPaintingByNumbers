import cv2
import pytesseract


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
    cv2.destroyAllWindows()


# if __name__ == "__main__":
#     capture_webcam("img/test.png")

pytesseract.pytesseract.tesseract_cmd = "D:\\tesseract\\tesseract.exe"
img = cv2.imread("img/2.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# print(pytesseract.image_to_string(img, lang="eng", config="--psm 6"))

# Detecting Characters
# hImg, wImg, _ = img.shape
# boxes = pytesseract.image_to_boxes(img, lang="eng", config="--psm 6")
# for b in boxes.splitlines():
#     # print(b)
#     b = b.split(" ")
#     # print(b)
#     x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
#     cv2.rectangle(img, (x, hImg - y), (w, hImg - h), (0, 0, 255), 1)
#     cv2.putText(img, b[0], (x, hImg - y + 25), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)

# Detecting Words
# hImg, wImg, _ = img.shape
# boxes = pytesseract.image_to_data(img, lang="eng", config="--psm 6")
# print(boxes)
# for x, b in enumerate(boxes.splitlines()):
#     # print(b)
#     if x != 0:
#         b = b.split()
#         print(b)
#         if len(b) == 12:
#             x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
#             cv2.rectangle(img, (x, y), (w + x, h + y), (0, 0, 255), 1)
#             cv2.putText(img, b[11], (x, y + 60), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)

# Detecting Numbers
hImg, wImg, _ = img.shape
cong = r"--oem 3 --psm 6 outputbase digits"
boxes = pytesseract.image_to_data(img, lang="eng", config=cong)
print(boxes)
for x, b in enumerate(boxes.splitlines()):
    # print(b)
    if x != 0:
        b = b.split()
        print(b)
        if len(b) == 12:
            x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
            cv2.rectangle(img, (x, y), (w + x, h + y), (0, 0, 255), 1)
            cv2.putText(img, b[11], (x - 5, y + 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)

cv2.imshow("Result", img)
cv2.waitKey(0)
