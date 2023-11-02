import cv2
import imgaug.augmenters as ia
import glob

images = []
images_path = glob.glob('./ConnectTheDots/*.jpeg')
for img_path in images_path:
    img = cv2.imread(img_path)
    images.append(img)

augmentation = ia.Sequential([
    ia.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, scale=(0.7, 1), rotate=(-30, 30)),
    ia.Multiply((0.8, 1.2)),
    ia.LinearContrast((0.6, 1.4)),
    ia.Sometimes(0.5, ia.GaussianBlur(sigma=(0, 0.5))),
])

target_images = 100

while len(images) < target_images:
    augmented_images = augmentation(images=images)
    for img in augmented_images:
        images.append(img)

images = images[:target_images]

for i, img in enumerate(images):
    cv2.imwrite(f'./ConnectTheDots/augmented_images/{i}.jpg', img)

# for img in augmented_images:
#     cv2.imshow('Augmented Image', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
