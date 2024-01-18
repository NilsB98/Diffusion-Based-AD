import os

from PIL import Image


def crop_image(image_path, output_path, left, top, right, bottom):
    image = Image.open(image_path)
    cropped_image = image.crop((left, top, right, bottom))
    cropped_image.save(output_path)


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith((".png", ".jpg", ".jpeg")): # You can add more formats if you want
            img = Image.open(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
    return images

def main():

    input_folder = r"C:\Users\nilsb\Desktop\206"
    output_folder = r"C:\Users\nilsb\Desktop\206\adjusted"

    percent = 50

    for filename in os.listdir(input_folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(os.path.join(input_folder, filename))

            width, height = img.size

            new_width = int(width * percent * 0.01)
            new_height = int(height * percent * 0.01)

            upper = new_height // 2 - 256 // 2 - 200
            left = new_width // 2 - 256 // 2
            right = new_width // 2 + 256 // 2
            bottom = new_height // 2 + 256 // 2 - 200

            img = img.resize((new_width, new_height))
            img = img.crop((left, upper, right, bottom))
            img.save(os.path.join(output_folder, filename))

            print(f"saved {filename} to {output_folder}")


if __name__ == '__main__':
    main()