import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import easygui
import tkinter as tk
from tkinter import filedialog

def reshape_image(screen_width, screen_height, image):
    image = cv2.resize(image, (screen_width, screen_height), interpolation=cv2.INTER_AREA)
    return image

def linear_image(image):
    image = np.reshape(image, (-1, 3))
    return image

def create_train_data(background, person):
    background = linear_image(background)
    person = linear_image(person)

    Xtrain = background
    Ytrain = np.zeros(len(background), dtype=np.uint8)
    Xtrain = np.concatenate((Xtrain, person))
    Ytrain = np.concatenate((Ytrain, np.ones(len(person), dtype=np.uint8)))

    return Xtrain, Ytrain

def get_prediction(model, test):
    test = linear_image(test)
    cat = model.predict(test)
    return cat

def show_predicted_image(image, width, height):
    image = np.reshape(image, (width, height))
    plt.figure()
    plt.imshow(image, cmap = 'gray')
    plt.show()

def get_mask(image, width, height):
    image = np.reshape(image, (height, width))
    return image

def train_model(background, person):
    Xtrain, Ytrain = create_train_data(background, person)
    start = time.time()
    print('Creating model ... ')
    # Here I am using LinearSVC --> faster results with training, but mostly with prediction
    model = LinearSVC()
    model.fit(Xtrain, Ytrain)
    print('Finished model ... ')
    end = time.time()
    seconds = end-start
    output = "Elapsed training time: " + str(int(seconds//60)) + "m " + str(int(seconds%60)) + "s"
    print(output)
    return model

def master_model(background_route, person_route, screen_width, screen_height, small_width, small_height):
    # get background image
    background = cv2.imread(background_route)
    background = reshape_image(small_width, small_height, background)

    # get person image
    person = cv2.imread(person_route)
    person = reshape_image(small_width, small_height, person)

    # manipulate train and test data
    model = train_model(background, person)

    return model

def get_new_image(model, test_image, new_background, screen_width, screen_height, small_width, small_height):
    width, height, _ = test_image.shape

    # get predicted image
    test_image = reshape_image(small_width, small_height, test_image)
    categories = get_prediction(model, test_image)
    mask = get_mask(categories, small_width, small_height)

    # create combined image
    final_image = np.copy(test_image)

    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j] == 0:
                final_image[i][j] = new_background[i][j]

    final_image = reshape_image(screen_width, screen_height, final_image)

    return final_image

def set_screen_dimensions(screen_width, screen_height, cam, percentage):
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)
    screen_width, screen_height = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    small_width, small_height = int(screen_width*percentage), int(screen_height*percentage)
    return screen_width, screen_height, small_width, small_height

def show_intro_instructions():
    # Introduction
    welcome_txt = "Hello there! This program will let you use a colored background as a 'green screen'. You will need" \
                  " to take 2 pictures and select one:\n\n1. Take image of only the colored background.\n2. Take image of yourself in a different background." \
                  "\n3. Select image of the new background you want.\n\nYou must take/choose the images in that specific order.\n\n" \
                  "Let's begin!"
    easygui.msgbox(welcome_txt, title="Project Introduction")

    # Choose where to save the taken images
    choose_destination_instructions = "In the following window you will pick the folder where you want to save the taken images..."
    easygui.msgbox(choose_destination_instructions, title="Person in different background", ok_button="Continue")

    root = tk.Tk()
    root.withdraw()
    save_images_route = filedialog.askdirectory()

    # First instruction
    first_img_instructions = "In the following camera window you will take the image of only the colored background...\nIn order to take the picture, press 'y' on " \
                             "your keyboard whenever you feel the picture is right."
    easygui.msgbox(first_img_instructions, title="Background only", ok_button="Continue")

    # Take first picture
    temp_cam = cv2.VideoCapture(0)
    while True:
        val, img = temp_cam.read()
        cv2.imshow('Background only', img)
        if cv2.waitKey(1) & 0xFF == ord('y'):
            background_route = save_images_route + '/background.jpg'
            cv2.imwrite(background_route, img)
            temp_cam.release()
            cv2.destroyAllWindows()
            break

    # Second instruction
    second_img_instructions = "In the following camera window you will take the image of yourself in a different background (from the previous one)...\nIn order to take the picture, press 'y' on " \
                             "your keyboard whenever you feel the picture is right."
    easygui.msgbox(second_img_instructions, title="Person in different background", ok_button="Continue")

    # Take second picture
    temp_cam = cv2.VideoCapture(0)
    while True:
        val, img = temp_cam.read()
        cv2.imshow('Background only', img)
        if cv2.waitKey(1) & 0xFF == ord('y'):
            person_route = save_images_route + '/person.jpg'
            cv2.imwrite(person_route, img)
            temp_cam.release()
            cv2.destroyAllWindows()
            break

    # Third instruction
    third_img_instructions = "In the following file chooser you will pick the image of the desired new background by itself..."
    easygui.msgbox(third_img_instructions, title="New background only", ok_button="Continue")

    # Choose third image
    root = tk.Tk()
    root.withdraw()
    new_background_route = filedialog.askopenfilename()

    start_txt = "Now we will train our model and show the results...\nThe terminal will let you know how the model is doing.\nWhenever you want to stop the recording, " \
                "just press 'q' on your keyboard."
    easygui.msgbox(start_txt, title="Starting model")

    return background_route, person_route, new_background_route

background_route, person_route, new_background_route = show_intro_instructions()
screen_width, screen_height = 680, 420
screen_percentage = 0.3

# Initialize the camera and its dimensions
cam = cv2.VideoCapture(0)
screen_width, screen_height, small_width, small_height = set_screen_dimensions(screen_width, screen_height, cam, screen_percentage)

# Get new background image
new_background = cv2.imread(new_background_route)
new_background = reshape_image(small_width, small_height, new_background)

# Every time the program starts we create the model
model = master_model(background_route, person_route, screen_width, screen_height, small_width, small_height)

# We get the new image for every frame and predict it in our model
while True:
    val, img = cam.read()
    img = get_new_image(model, img, new_background, screen_width, screen_height, small_width, small_height)
    cv2.imshow('Changed background', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
