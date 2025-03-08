import cv2
from matplotlib import pyplot as plt


# opening image
img = cv2.imread("image.jpg")

# OpenCV opens images as BRG
# but we want RGB and grayscale

img_gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
img_rgb = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

banana_data = cv2.CascadeClassifier("banana_cascade.xml")
found = banana_data.detectMultiScale(img_gray, minSize =(30,30))

amount_found = len(found)

if amount_found != 0:

    # We draw a green rectangle around recognized sign

    for(x,y,width,height) in found:
        cv2.rectangle(img_rgb,(x,y), x + width, y + height, (0,255,0), 5)

    # Creates the environment of 
    # the picture and shows it
    plt.subplot(1, 1, 1)
    plt.imshow(img_rgb)
    plt.show()

    # Still image save