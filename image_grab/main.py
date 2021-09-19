import cv2
import numpy as np
from PIL import ImageGrab

screen = ImageGrab.grab()

image = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)

# Save image
cv2.imwrite('screenshot.png', image)

# Show image
cv2.imshow('window', image)

cv2.waitKey(0)
