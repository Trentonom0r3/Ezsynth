import cv2
import numpy as np

def generate_gradient(img):
    bgr_channels = cv2.split(img)
    grad_x = []
    grad_y = []
    for channel in bgr_channels:
        grad_x_channel = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3, scale=1/8.0, borderType=cv2.BORDER_REFLECT)
        grad_y_channel = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3, scale=1/8.0, borderType=cv2.BORDER_REFLECT)
        grad_x.append(grad_x_channel)
        grad_y.append(grad_y_channel)
    gradient_x = cv2.merge(grad_x)
    gradient_y = cv2.merge(grad_y)
    return gradient_x, gradient_y

def assemble_min_error_gradient(grad_x_a, grad_y_a, grad_x_b, grad_y_b, err_mask):
    grad_x_min = grad_x_a.copy()  # Initialize with grad_x_a
    grad_y_min = grad_y_a.copy()  # Initialize with grad_y_a
    # Copy grad_a's gradient using mask
    grad_x_min[err_mask == 1] = grad_x_a[err_mask == 1]
    grad_y_min[err_mask == 1] = grad_y_a[err_mask == 1]
    # Flip the mask
    flip_err_mask = 1 - err_mask
    # Copy grad_b's gradient using flipped mask
    grad_x_min[flip_err_mask == 1] = grad_x_b[flip_err_mask == 1]
    grad_y_min[flip_err_mask == 1] = grad_y_b[flip_err_mask == 1]
    return grad_x_min, grad_y_min

def gradient_blending(images_a, images_b, err_masks):
    out_blend_x = []
    out_blend_y = []
    for img_a, img_b, err_mask in zip(images_a, images_b, err_masks):
        err_mask[err_mask > 0] = 1
        grad_x_a, grad_y_a = generate_gradient(img_a)
        grad_x_b, grad_y_b = generate_gradient(img_b)
        grad_x_min, grad_y_min = assemble_min_error_gradient(grad_x_a, grad_y_a, grad_x_b, grad_y_b, err_mask)
        out_blend_x.append(grad_x_min)
        out_blend_y.append(grad_y_min)
    return out_blend_x, out_blend_y
