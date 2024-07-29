import cv2


class Cartoonizer:
    """Cartoonizer effect
    A class that applies a cartoon effect to an image or video feed.
    The class uses a bilateral filter and adaptive thresholding to create
    a cartoon effect.
    """

    def __init__(self):
        pass

    def render(self, img_rgb):
        # Resize the frame for consistent processing
        img_rgb = cv2.resize(img_rgb, (1366, 768))
        numDownSamples = 2  # number of downscaling steps
        numBilateralFilters = 50  # number of bilateral filtering steps

        # -- STEP 1 --

        # Downsample image using Gaussian pyramid
        img_color = img_rgb
        for _ in range(numDownSamples):
            img_color = cv2.pyrDown(img_color)

            # Repeatedly apply small bilateral filter instead of applying
        # one large filter
        for _ in range(numBilateralFilters):
            img_color = cv2.bilateralFilter(img_color, 9, 9, 7)

            # Upsample image to original size
        for _ in range(numDownSamples):
            img_color = cv2.pyrUp(img_color)

            # -- STEPS 2 and 3 --
        # Convert to grayscale and apply median blur
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 3)

        # -- STEP 4 --
        # Detect and enhance edges
        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 9, 2)

        # -- STEP 5 --
        # Convert back to color so that it can be bit-ANDed with color image
        (x, y, z) = img_color.shape
        img_edge = cv2.resize(img_edge, (y, x))
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)

        return cv2.bitwise_and(img_color, img_edge)

    # Initialize the Cartoonizer object


cartoonizer = Cartoonizer()

# Open the camera
cap = cv2.VideoCapture(0)  # 0 for default camera

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab a frame.")
        break

    # Apply cartoon effect
    cartoon_frame = cartoonizer.render(frame)

    # Display the resulting frame
    cv2.imshow('Cartoon Version', cartoon_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
