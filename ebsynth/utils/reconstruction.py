import numpy as np
import cv2
from scipy.fftpack import fft2, ifft2

class ScreenedPoissonSolver:
    def __init__(self, image_path, grad_x_path=None, grad_y_path=None):
        # Read the original image
        self.image = image_path
        
        # Read the gradients if provided, else initialize to None
        self.gradient_x = grad_x_path if grad_x_path else None
        self.gradient_y = grad_y_path if grad_y_path else None
        
        # Initialize solution to None
        self.solution = None
         
    def solve_equation(self, alpha=1.0):
        # Initialize empty array for the solution
        self.solution = np.zeros_like(self.image)
        
        # Solve the Screened Poisson Equation for each channel
        for ch in range(3):
            self.fft_image = fft2(self.image[:, :, ch])
            self.fft_gradient_x = fft2(self.gradient_x[:, :, ch]) if self.gradient_x is not None else 0
            self.fft_gradient_y = fft2(self.gradient_y[:, :, ch]) if self.gradient_y is not None else 0
            
            rows, cols = self.image.shape[:2]
            
            freq_x = np.fft.fftfreq(cols)
            freq_y = np.fft.fftfreq(rows)
            freq_x, freq_y = np.meshgrid(freq_x, freq_y)
            
            denominator = (4.0 - 2.0*(np.cos(2.0*np.pi*freq_x) + np.cos(2.0*np.pi*freq_y))) + alpha
            fft_solution = (self.fft_image + self.fft_gradient_x + self.fft_gradient_y) / denominator
            
            # Inverse FFT to get the solution in the spatial domain
            solution_channel = np.real(ifft2(fft_solution))
            
            # Normalize the solution to be in [0, 255]
            solution_channel = np.clip(solution_channel, 0, 255)
            
            # Convert to uint8
            self.solution[:, :, ch] = solution_channel.astype(np.uint8)
    
    def show_result(self):
        # Display the resulting image
        cv2.imshow('Result', self.solution)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def compute_gradients_from_image(self):
        # Compute gradients in x and y directions
        self.image_gradient_x = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=1, scale=1/8.0, borderType=cv2.BORDER_REFLECT)
        self.image_gradient_y = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=1, scale=1/8.0, borderType=cv2.BORDER_REFLECT)

    def mix_gradients(self, weight=0.5):
        self.gradient_x = weight * self.gradient_x + (1 - weight) * self.image_gradient_x
        self.gradient_y = weight * self.gradient_y + (1 - weight) * self.image_gradient_y

    def run(self, alpha=1.0, gradient_mix_weight=0.05):
        self.compute_gradients_from_image()
        self.mix_gradients(gradient_mix_weight)
        self.solve_equation(alpha)
        return self.solution


    
