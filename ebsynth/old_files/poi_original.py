from numba import njit
import numpy as np
import pyamg
import cv2
import time

@njit(fastmath=True)
def build_b(y_min, y_max, x_min, x_max, index_map, source_channel, destination_channel, mask):
    b = np.zeros((y_max - y_min + 1) * (x_max - x_min + 1))
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            index = index_map[y, x]

            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                    neighbor_index = index_map[ny, nx]
                    if neighbor_index < 0:
                        b[index] += destination_channel[ny, nx]

            b[index] += source_channel[y, x] * 4
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < source_channel.shape[0] and 0 <= nx < source_channel.shape[1]:
                    b[index] -= source_channel[ny, nx]

    return b

@njit
def build_csr_data(y_min, y_max, x_min, x_max, index_map, source_channel, destination_channel, mask):
    
    b = build_b(y_min, y_max, x_min, x_max, index_map, source_channel, destination_channel, mask)
    return b


def poisson_blend_single_channel(source_channel, destination_channel, mask, index_map):
    #source_channel = source_channel.astype(np.float64)
    #destination_channel = destination_channel.astype(np.float64)

    y_indices, x_indices = np.where(mask != 0)
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    region_height, region_width = y_max - y_min + 1, x_max - x_min + 1
    num_pixels = region_height * region_width

    index_map[y_min:y_max + 1, x_min:x_max + 1] = np.arange(num_pixels).reshape((region_height, region_width))
    
    start0 = time.time()  # timer for build_csr_data (which is the slowest part of the function)
    
    b = build_csr_data(y_min, y_max, x_min, x_max, index_map, source_channel, destination_channel, mask)
    
    end0 = time.time()
    print("Poisson blending csr setup (internal) took {} seconds.".format(end0 - start0))

    A_csr = pyamg.gallery.poisson((region_height, region_width), format='csr')  # tested multiple solvers, this one is the fastest


    #A_csr = csr_matrix((data, (rows, cols)), shape=(num_pixels, num_pixels))
    #ml = pyamg.classical.classical.ruge_stuben_solver(A_csr)
    start2 = time.time() # timer for ml.solve
    
    ml = pyamg.ruge_stuben_solver(A_csr)  # tested multiple solvers, this one is the fastest
    x = ml.solve(b, tol=1e-3, x0=np.zeros_like(b), maxiter = 50, accel='cg')
    
    end2 = time.time()
    print("Poisson blending solving (internal) took {} seconds.".format(end2 - start2))

    blended_region = x.reshape((region_height, region_width))

    result = destination_channel.copy()
    result[y_min:y_max + 1, x_min:x_max + 1] = blended_region
    
    return result

from concurrent.futures import ProcessPoolExecutor

def blend_channel(args):
    channel, source, destination, mask = args
    source_channel = source[:, :, channel].astype(np.float64)
    destination_channel = destination[:, :, channel].astype(np.float64)
    index_map = -np.ones_like(mask, dtype=np.int32)
    return poisson_blend_single_channel(source_channel, destination_channel, mask, index_map)

def poisson_blend(source, destination, mask):
    blended_image = np.zeros_like(destination)

    blended_channels = [(channel, source, destination, mask) for channel in range(3)]
    for channel, blended_channel in enumerate(blended_channels):
        blended_image[:, :, channel] = blended_channel

    return blended_image


def main(source, destination, mask):
    # Start a timer
    start = time.time()

    # Perform poisson blending
    blended_image = poisson_blend(source, destination, mask)

    # Ensure that the output of poisson_blend has the same shape as the destination image
    assert blended_image.shape == destination.shape

    # End the timer, update progress
    end = time.time()
    print("Poisson blending took {} seconds.".format(end - start))

    # Display the blended image
    cv2.imshow("blended_image", blended_image)
    cv2.imshow("source", source)
    cv2.imshow("destination", destination)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    source = cv2.imread("C:/Users/tjerf/Desktop/Testing/src/Testvids/Output/forward/style_forward_10.png")
    destination = cv2.imread("C:/Users/tjerf/Desktop/Testing/src/Testvids/Output/backward/style_backward_10.png")
    mask = cv2.imread("C:/Users/tjerf/Desktop/Testing/src/Testvids/Output/masks/selection_mask_10.png", cv2.IMREAD_GRAYSCALE)
    poisson_blend(source, destination, mask)
    

    # TIMING RESULTS
    #Poisson blending csr setup (internal) took 0.8722474575042725 seconds.
    #Poisson blending csr setup (internal) took 0.8564944267272949 seconds.
    #Poisson blending csr setup (internal) took 0.8655593395233154 seconds.
    #Poisson blending solving (internal) took 0.935208797454834 seconds.
    #Poisson blending solving (internal) took 0.9332070350646973 seconds.
    #Poisson blending solving (internal) took 0.948847770690918 seconds.
    #Poisson blending took 3.0318758487701416 seconds.