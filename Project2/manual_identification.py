# Step 1: Import the required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Load the images that need to be aligned and stitched
# Make sure the images are in the same directory or specify the path
img1 = cv2.imread('1.jpeg')  # Replace with the filename of your first image
img2 = cv2.imread('2.jpeg')  # Replace with the filename of your second image
img3 = cv2.imread('3.jpeg')  # Replace with the filename of your third image

# Convert images from BGR (OpenCV default) to RGB for displaying in matplotlib
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img3_rgb = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

# Step 3: Display each image separately to select corresponding points

# Select points in the first image
plt.figure(figsize=(10, 8))
plt.imshow(img1_rgb)
plt.title("Select points in Image 1")
points_img1 = plt.ginput(n=-1, timeout=0)  # Select any number of points
plt.close()

# Select points in the second image
plt.figure(figsize=(10, 8))
plt.imshow(img2_rgb)
plt.title("Select points in Image 2")
points_img2 = plt.ginput(n=-1, timeout=0)  # Select corresponding points in the second image
plt.close()

# Select points in the third image
plt.figure(figsize=(10, 8))
plt.imshow(img3_rgb)
plt.title("Select points in Image 3")
points_img3 = plt.ginput(n=-1, timeout=0)  # Select corresponding points in the third image
plt.close()

# Step 4: Display the selected points to verify
print("Selected points in Image 1:", points_img1)
print("Selected points in Image 2:", points_img2)
print("Selected points in Image 3:", points_img3)

# Convert the lists of points to NumPy arrays for later use
points_img1 = np.array(points_img1)
points_img2 = np.array(points_img2)
points_img3 = np.array(points_img3)

# Step 5: Save or print the points for reference
# Optionally save the points to a file if needed later
np.save('points_img1.npy', points_img1)
np.save('points_img2.npy', points_img2)
np.save('points_img3.npy', points_img3)
