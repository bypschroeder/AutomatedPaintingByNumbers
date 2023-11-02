import cv2
import pytesseract
import matplotlib.pyplot as plt


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
img_file = "img/3.jpg"
img = cv2.imread(img_file)


# https://stackoverflow.com/questions/28816046/displaying-different-images-with-actual-size-in-matplotlib-subplot
def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)
    height = im_data.shape[0]
    width = im_data.shape[1]

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()


# display(img_file)

# Inverting Image
inverted_img = cv2.bitwise_not(img)
cv2.imwrite("img/inverted.jpg", inverted_img)
# display("img/inverted.jpg")

# Rescaling


# Binarization
def grayscale(image):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


gray_img = grayscale(img)
cv2.imwrite("img/gray.jpg", gray_img)
# display("img/gray.jpg") cant display
thresh, im_bw = cv2.threshold(gray_img, 110, 255, cv2.THRESH_BINARY)
cv2.imwrite("img/binary.jpg", im_bw)

display("img/binary.jpg")

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
cock = cv2.imread('img/no_noise.jpg')
hImg, wImg, _ = cock.shape
cong = r"--oem 3 --psm 6 outputbase digits"
boxes = pytesseract.image_to_data(cock, lang="eng", config=cong)
print(boxes)
for x, b in enumerate(boxes.splitlines()):
    # print(b)
    if x != 0:
        b = b.split()
        print(b)
        if len(b) == 12:
            x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
            cv2.rectangle(cock, (x, y), (w + x, h + y), (0, 0, 255), 1)
            cv2.putText(cock, b[11], (x - 5, y + 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)

cv2.imshow("Result", cock)
cv2.waitKey(0)
