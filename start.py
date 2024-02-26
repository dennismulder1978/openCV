import cv2
import numpy as np

def main():
    # Load the image
    img = cv2.imread('2.png')
    if img is None:
        print("Error: Unable to load image.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny edge detector
    edges = cv2.Canny(blur, 120, 200)

    # Find contours from the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(contours)
    # Draw all contours on the original image
    cv2.drawContours(img, contours, -1, (15, 15, 255), 3)

    # Display the image with contours
    cv2.imshow('Contours', img)
    cv2.imwrite('2a.png', img)
    # Wait for a key press to exit
    cv2.waitKey(0)

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
