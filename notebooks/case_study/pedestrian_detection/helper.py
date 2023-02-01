import cv2
import matplotlib.pyplot as plt

def draw(title, image, reverse = False):
    plt.title(title)
    plt.axis("off")
    if reverse == True:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image)
    plt.show()
