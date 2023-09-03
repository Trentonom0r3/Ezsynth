import cv2
import numpy as np
import scipy.fftpack


def fourier_solve(img_data, img_grad_x, img_grad_y, data_cost):
    img_data = img_data.astype(float)
    img_grad_x = img_grad_x.astype(float)
    img_grad_y = img_grad_y.astype(float)

    gx = img_grad_x.reshape(-1)
    gy = img_grad_y.reshape(-1)


    node_count = img_data.shape[0] * img_data.shape[1]
    fft_buff = np.empty(node_count, dtype=float)

    ft_lap_y = -4.0 + 2.0 * np.cos(np.pi * np.arange(img_data.shape[0]) / (img_data.shape[0] - 1))
    ft_lap_x = 2.0 * np.cos(np.pi * np.arange(img_data.shape[1]) / (img_data.shape[1] - 1))

    fft_buff = scipy.fftpack.idctn(fft_buff.reshape(img_data.shape[:2]), type=1, norm='ortho').reshape(-1)

    for i_channel in range(img_data.shape[2]):
        node_addr = 0
        pixel_addr = i_channel
        right_pixel_addr = img_data.shape[2] + i_channel
        top_pixel_addr = img_data.shape[1] * img_data.shape[2] + i_channel

        dc_sum = 0.0

        for y in range(img_data.shape[0]):
            for x in range(img_data.shape[1]):
                dc_mult = 1.0
                if 0 < x < img_data.shape[1] - 1:
                    dc_mult *= 2.0
                if 0 < y < img_data.shape[0] - 1:
                    dc_mult *= 2.0
                dc_sum += dc_mult * img_data[y, x, i_channel]

                fft_buff[node_addr] = data_cost * img_data[y, x, i_channel]

                if 0 < x < img_data.shape[1] - 1:
                    fft_buff[node_addr] -= (gx[right_pixel_addr] - gx[pixel_addr])
                else:
                    fft_buff[node_addr] -= (-2.0 * gx[pixel_addr])

                if 0 < y < img_data.shape[0] - 1:
                    fft_buff[node_addr] -= (gy[top_pixel_addr] - gy[pixel_addr])
                else:
                    fft_buff[node_addr] -= (-2.0 * gy[pixel_addr])

                node_addr += 1
                pixel_addr += img_data.shape[2]
                right_pixel_addr += img_data.shape[2]
                top_pixel_addr += img_data.shape[2]


        fft_buff = scipy.fftpack.dct(fft_buff.reshape(img_data.shape[:2]), type=1, norm='ortho').reshape(-1)

        node_addr = 0
        for y in range(img_data.shape[0]):
            for x in range(img_data.shape[1]):
                ft_lap_response = ft_lap_y[y] + ft_lap_x[x]
                fft_buff[node_addr] /= (data_cost - ft_lap_response)
                node_addr += 1

        fft_buff[0] = dc_sum

        f_hat = scipy.fftpack.idct(fft_buff, type=1, norm='ortho').reshape(img_data.shape[:2])
        
        # Scale and map f_hat values to 0-255 range
        f_hat_min = np.min(f_hat)
        f_hat_max = np.max(f_hat)
         # Apply a scaling strategy that keeps values in a visible range
        scaled_f_hat = (f_hat - np.min(f_hat)) / (np.max(f_hat) - np.min(f_hat)) * 255
    
        
        # Convert to uint8
        f_hat_uint8 = scaled_f_hat.astype(np.uint8)
        
        img_data[:, :, i_channel] = f_hat_uint8
    print("\tDone.")
    return img_data

def poisson_blend(hp_blends, grad_x, grad_y):
    final_blends = []
    data_cost = 0.1
    
    for i in range(len(hp_blends)):
        current_frame = hp_blends[i].astype(np.float32)
        grad_x_i = grad_x[i].astype(np.float32)
        grad_y_i = grad_y[i].astype(np.float32)

        img = fourier_solve(current_frame, grad_x_i, grad_y_i, data_cost)    
        final_blends.append(img.astype(np.uint8))
    
    return final_blends


