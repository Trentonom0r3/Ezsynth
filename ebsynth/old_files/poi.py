from numba import njit, prange
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
                    b[index] -= source_channel[ny, nx]

            b[index] += source_channel[y, x] * 4


    return b

@njit
def build_csr_data(y_min, y_max, x_min, x_max, index_map, source_channel, destination_channel, mask):
    
    b = build_b(y_min, y_max, x_min, x_max, index_map, source_channel, destination_channel, mask)
    return b


def poisson_blend_single_channel(source_channel, destination_channel, mask, index_map, y_min, y_max, x_min, x_max):
    region_height, region_width = y_max - y_min + 1, x_max - x_min + 1
    
    b = build_csr_data(y_min, y_max, x_min, x_max, index_map, source_channel, destination_channel, mask)


    A_csr = pyamg.gallery.poisson((region_height, region_width), format='csr')  # tested multiple solvers, this one is the fastest

    ml = pyamg.ruge_stuben_solver(A_csr)  # tested multiple solvers, this one is the fastest
    x = ml.solve(b, tol=1e-2, x0=np.zeros_like(b), maxiter=25, accel='cg')


    blended_region = x.reshape((region_height, region_width))
    result = destination_channel.copy()
    result[y_min:y_max + 1, x_min:x_max + 1] = blended_region
    
    return result


from concurrent.futures import ProcessPoolExecutor

def blend_channel(args):
    channel, source, destination, mask, y_min, y_max, x_min, x_max, index_map = args
    source_channel = source[:, :, channel].astype(np.float32)
    destination_channel = destination[:, :, channel].astype(np.float32)
    return poisson_blend_single_channel(source_channel, destination_channel, mask,
                                        index_map, y_min, y_max, x_min, x_max)

def poisson_blend(source, destination, mask):
    blended_image = np.zeros_like(destination)

    y_indices, x_indices = np.where(mask != 0)
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()
    index_map = -np.zeros_like(mask, dtype=np.int32)
    index_map[y_min:y_max + 1, x_min:x_max + 1] = np.arange((y_max - y_min + 1) * (x_max - x_min + 1)).reshape((y_max - y_min + 1, x_max - x_min + 1))

    blended_channels = map(blend_channel, [(channel, source, destination, mask, 
                                                         y_min, y_max, x_min, x_max, index_map) for channel in range(3)])
    for channel, blended_channel in enumerate(blended_channels):
        blended_image[:, :, channel] = blended_channel

    return blended_image


"""
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
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    source = cv2.imread("C:/Users/tjerf/Desktop/Testing/src/Testvids/Output/forward6.png")
    destination = cv2.imread("C:/Users/tjerf/Desktop/Testing/src/Testvids/Output/backward6.png")
    mask = cv2.imread("C:/Users/tjerf/Desktop/Testing/src/Testvids/Output/selection6.png", cv2.IMREAD_GRAYSCALE)
    main(source, destination, mask)
"""