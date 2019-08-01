import cv2
# only for np.asarray() function
import numpy as np

# used by conv function to do element wise multiplication
def elmwise_mul(a,b):
    c = []
    for i in range(0,len(a)):
        temp=[]
        for j in range(0,len(a[0])):
            temp.append(a[i][j] * b[i][j])
        c.append(temp)
    return c

# used to flatten multi dimensional array to single dimension to find sum, max, min etc.
def flatten_arr(arr):
    result = []
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            result.append(arr[i][j])
    return result

# convolve image with a kernel of size kernel_size, stride of 1
def conv(img, kernel, kernel_size):        
    result = []
    # flips the kernel for convolution
    kernel = kernel[::-1][::-1]
    for i in range(img.shape[0]):
        if i <= img.shape[0] - kernel_size:
            row = []
            for j in range(img.shape[1]):
                if j <= img.shape[1] - kernel_size:
                    # element wise multiplication with kernel and calculates sum and appends to the result
                    img_arr = np.asarray(img[i:i+kernel_size, j:j+kernel_size])
                    temp = elmwise_mul(kernel, img_arr)
                    temp = sum(flatten_arr(temp))
                    row.append(temp)
            result.append(row)
    return result

def normalize_and_display(result, window_name):
    # flatten the array to calculate maximum and minimum 
    arr = flatten_arr(result)
    min_val = min(arr)
    max_val = max(arr)

    # normalize the resultant image before displaying it
    result = (result - min_val) / (max_val - min_val)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, result)
    cv2.waitKey(0)

if __name__ == "__main__":        
    img = cv2.imread("./data/proj1-task1.png", 0)

    # kernel used to detect horizontal edges
    sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

    # kernel used to   detect vertical edges
    sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    # convolves the kernel with image to detect horizontal edges 
    horiz_result = conv(img, sobel_x, 3)

    # normalizes image, displays it and saves image to disk as ./Horiz_conv.png
    normalize_and_display(horiz_result, "Horiz_conv")

    # convolves the kernel with image to detect vertical edges 
    vert_result = conv(img, sobel_y, 3)

    # normalizes image, displays it and saves image to disk as ./Vert_conv.png
    normalize_and_display(vert_result, "Vert_conv")