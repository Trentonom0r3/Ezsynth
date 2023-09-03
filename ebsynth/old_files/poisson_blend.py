import numpy as np
import cv2
from scipy.fftpack import dct, idct  # You can continue to use pyfftw if you have it installed

def poisson_blend(img_data, img_grad_x, img_grad_y, data_cost):
    rows, cols, channels = img_data.shape

    # Precompute Fourier Laplacian
    ft_lap_x = 2.0 * np.cos(np.pi * np.arange(cols) / (cols - 1)) - 2.0
    ft_lap_y = 2.0 * np.cos(np.pi * np.arange(rows) / (rows - 1)) - 2.0
    ft_lap_response = ft_lap_y[:, np.newaxis] + ft_lap_x[np.newaxis, :]

    # Initialize variables
    fft_buff = np.ones((rows, cols, channels), dtype=np.float32)

    # Calculate DCT of data term
    fft_buff = dct(dct(data_cost * img_data, axis=0, type=2), axis=1, type=2)

    # Calculate DCT of gradient terms
    for i_channel in range(channels):
        gx = img_grad_x[:, :, i_channel].flatten()
        gy = img_grad_y[:, :, i_channel].flatten()

        # Create matrices for gradient terms
        grad_x_matrix = np.zeros((rows, cols), dtype=np.float32)
        grad_y_matrix = np.zeros((rows, cols), dtype=np.float32)

        # Fill in gradient terms (assuming gx and gy are already appropriately scaled)
        grad_x_matrix[:, 1:] += gx.reshape(rows, cols)[:, 1:]
        grad_x_matrix[:, :-1] -= gx.reshape(rows, cols)[:, :-1]
        grad_y_matrix[1:, :] += gy.reshape(rows, cols)[1:, :]
        grad_y_matrix[:-1, :] -= gy.reshape(rows, cols)[:-1, :]

        # Add gradient terms to fft_buff
        fft_buff[:, :, i_channel] -= dct(dct(grad_x_matrix + grad_y_matrix, axis=0, type=2), axis=1, type=2)

    # Solve in Fourier domain
    fft_buff /= (data_cost - ft_lap_response)[:, :, np.newaxis]

    # Inverse DCT to obtain the image
    img_data = idct(idct(fft_buff, axis=0, type=2), axis=1, type=2) / (4.0 * (cols - 1) * (rows - 1))
    
    return np.clip(img_data, 0, 255).astype(np.uint8)

def adaptive_data_cost(grad_x, grad_y):
    # Compute the magnitude of the gradient
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    
    # Compute a basic measure of 'edginess' in the image
    mean_magnitude = np.mean(grad_magnitude)
    
    # Adapt the data cost based on the mean gradient magnitude
    return 0.1 + 0.9 * np.tanh(mean_magnitude / 30.0)

# Debugging and example usage
img = cv2.imread("C:/Users/tjerf/Desktop/Testing/src/Testvids/Output/forward6.png")
grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0)
grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1)

# Calculate adaptive data cost
data_cost = adaptive_data_cost(grad_x, grad_y)

# Run the Fourier solver
blended_image = poisson_blend(img, grad_x, grad_y, data_cost)

# Visualize the blended image
cv2.imshow("Blended Image", blended_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

