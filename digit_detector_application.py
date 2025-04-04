import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

model = load_model("bestmodel.h5")

window_size_x = 640
window_size_y = 480

# Define boundary padding for drawn digits
boundary_pixels = 7

white = (255,255,255)   # Drawing color
black = (0,0,0)         # Background color
red = (255,0,0)         # Prediction text and rectangle color

digit_labels = {0:"Zero", 1:"One", 2:"Two", 3:"Three", 4:"Four", 5:"Five", 6:"Six", 7:"Seven", 8:"Eight", 9:"Nine"}

# Initialize pygame and set up the display
pygame.init()
font = pygame.font.SysFont('arial', size = 20)                      # Font for displaying predictions
surface = pygame.display.set_mode((window_size_x, window_size_y))   # Create drawing window
pygame.display.set_caption("Digit Board")                           # Window title

iswriting = False       # Tracks if the mouse is drawing
x_coordinates = []      # Stores x positions of drawn points
y_coordinates = []      # Stores y positions of drawn points
cansave_img = False     # Flag to save the drawn image
can_predict = True      # Flag to enable/disable predictions


# Main loop to handle events and drawing
while True:

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        elif event.type == MOUSEMOTION and iswriting: # Draw circles where the mouse moves while clicking
            x, y = event.pos
            pygame.draw.circle(surface, white, (x,y), 4,0) # Draw a small white circle
            x_coordinates.append(x)
            y_coordinates.append(y)

        elif event.type == MOUSEBUTTONDOWN:
            iswriting = True
        elif event.type == MOUSEBUTTONUP:
            iswriting = False
            x_coordinates = sorted(x_coordinates)
            y_coordinates = sorted(y_coordinates)
            # Calculate bounding box around the drawn digit with padding
            rectangle_x_min, rectangle_x_max = max(x_coordinates[0]-boundary_pixels, 0), min(window_size_x, x_coordinates[-1]+boundary_pixels)
            rectangle_y_min, rectangle_y_max = max(y_coordinates[0]-boundary_pixels, 0), min(window_size_x, y_coordinates[-1]+boundary_pixels)
            # Reinitialization
            x_coordinates = []
            y_coordinates = []
            # Extract the drawn area as a numpy array
            image_array = np.array(pygame.PixelArray(surface))[rectangle_x_min:rectangle_x_max, rectangle_y_min:rectangle_y_max].T.astype(np.float32)

            
            if can_predict:
                img = cv2.resize(image_array, (28,28)) # (28,28) since the model was trained with this size
                img = np.pad(img, (10,10), 'constant', constant_values=0)
                img = cv2.resize(img, (28,28))/255 # white = (255,255,255), normalizing to range [0-1]
                
                cansave_img = False
                digit = str(digit_labels[np.argmax(model.predict(img.reshape(1,28,28,1)))]) # Predict the digit and get its label

                if cansave_img:
                    cv2.imwrite('image.png'.format(np.argmax(model.predict(img.reshape(1,28,28,1)))),img)

                textsurf = font.render(digit, True, red)
                pygame.draw.rect(surface,red, pygame.Rect(rectangle_x_min, rectangle_y_min, rectangle_x_max-rectangle_x_min, rectangle_y_max-rectangle_y_min), 2)

                surface.blit(textsurf, (rectangle_x_min, rectangle_y_min-18))

            if event.type == KEYDOWN:
                if event.key== pygame.K_BACKSPACE:
                    surface.fill(black)
        
        pygame.display.update() # Update the display after each event