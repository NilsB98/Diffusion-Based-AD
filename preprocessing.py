# Preprocessing to increase visibility of cracks.
import os

import matplotlib.pyplot as plt
import cv2
import numpy as np


def show_images_side_by_side(image1, image2, image3, plt_title, title1="original", title2="transformed"):
    fig, ax = plt.subplots(1, 3)
    fig.set_figwidth(30)
    fig.set_figheight(24)

    # Turn off the axes
    ax[0].axis('off')
    ax[1].axis('off')

    # Display the images
    ax[0].imshow(image1, cmap='gray')
    ax[0].set_title(title1)

    ax[1].imshow(image2, cmap='gray')
    ax[1].set_title(title2)

    ax[2].imshow(image3, cmap='gray')
    ax[2].set_title(title2)

    plt.suptitle(plt_title)
    plt.show()



if __name__ == '__main__':
    dir_path = r"C:/Users/Nils.Braehmer/Pictures/rissbilder"
    img_names = os.listdir(dir_path)


    for img_name in img_names:
        img = cv2.imread(f"{dir_path}/{img_name}", cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (800, 600))

        pos_number = img_name[31:35]
        core_number = img_name[36:40]

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(img)

        bright_cont = cv2.convertScaleAbs(cl1, alpha=1.5, beta=0)

        sobel = cv2.Sobel(cl1, cv2.CV_64F, 1, 1, ksize=5)

        equ = cv2.equalizeHist(img)
        # res = np.hstack((cl1, sobel))  # stacking images side-by-side
        sobel_8u = cv2.convertScaleAbs(sobel)

        # cv2.imshow("clahe", cl1)
        cv2.imshow("original", img)
        cv2.imshow("clahe, brightness, contrast", bright_cont)
        cv2.waitKey(0)

        # plt.imshow(res)
        # plt.show()

        # show_images_side_by_side(img, cl1, sobel, f"{core_number} - {pos_number}")
