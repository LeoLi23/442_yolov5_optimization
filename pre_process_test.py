import cv2
import numpy as np
import os
# import cv2.ximgproc

# Grey World Assumption
def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)

def laplacian_weight_map(img):
    # ASSUME RGB IMAGE IS PASSED IN.
    img = cv2.cvtColor(src=img, code=cv2.COLOR_RGB2LAB)
    return np.abs(cv2.Laplacian(src=img, ksize=3))


def saliency_weight_map(img):  
    # Convert image to grayscale
    if(len(img.shape) > 2):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img = img
        
    # Apply Gaussian Smoothing
    gaussian = cv2.GaussianBlur(img,(5, 5),0) 
        
    # Apply Mean Smoothing
    img_mean = np.mean(img)
        
    # Generate Saliency Map
    saliencymap = np.absolute(gaussian - img_mean)
    return saliencymap

def preprocess_image(img):
    
    # white balance
    img = white_balance(img)

    # histogram equalization 
    for i in range(3):
        img[:,:,i] = cv2.equalizeHist(img[:,:,i])
    
    img2_lab = cv2.cvtColor(src=img, code=cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(2, 2))
    img2_lab[:, :, 0] = clahe.apply(img2_lab[:, :, 0])
    img = cv2.cvtColor(src=img2_lab, code=cv2.COLOR_LAB2BGR)

    # 4. Color Correction
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 1] = cv2.multiply(img_hsv[:, :, 1], 1.5)
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    # 5. Image Dehazing
    img = white_balance(img)
    #img = cv2.ximgproc.createSimpleWB().balanceWhite(img)
    # print(f"Before sharpen: {img[0,0,:]}")
    
    # 6. Sharpening
    # kernel = np.array([[-1, -1, -1],
    #                    [-1, 9, -1],
    #                    [-1, -1, -1]])
    # img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    # print(f"After sharpen: {img[0,0,:]}")

    # 6. Subtle Sharpening
    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3, -0.5],
                       [0, -0.5, 0]])
    img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    print(f"After sharpen: {img[0,0,:]}")
    
    
    # for i in range(3):
    #     img[:,:,i] = clahe.apply(img[:,:,i])
    
    # img2_l_clahe = clahe.apply(img2_lab[:,:,0])
    # img2_lab[:,:,0] = img2_l_clahe
    # img2 = cv2.cvtColor(src=img2_lab, code=cv2.COLOR_LAB2RGB)

    return img

    # # Sharpen and apply histogram equalization
    # kernel = np.array([[0, -1, 0],
    #                 [-1, 5,-1],
    #                 [0, -1, 0]])
    # img1 = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    # img1 = cv2.equalizeHist(src=img1)

if __name__ == '__main__':
    filename_1 = 'test_img.jpg'
    filename_2 = 'test_img_2.jpg'
    filename_3 = 'test_img_3.jpg'
    img_1 = cv2.imread(os.getcwd() + "/" + filename_1)
    img_2 = cv2.imread(os.getcwd() + "/" + filename_2)
    img_3 = cv2.imread(os.getcwd() + "/" + filename_3)
    img_processed_1 = preprocess_image(img_1)
    img_processed_2 = preprocess_image(img_2)
    img_processed_3 = preprocess_image(img_3)

    cv2.imwrite(os.getcwd() + "/" + 'test_img_output_1.jpg', img_processed_1)
    cv2.imwrite(os.getcwd() + "/" +'test_img_output_2.jpg', img_processed_2)
    cv2.imwrite(os.getcwd() + "/" + 'test_img_output_3.jpg', img_processed_3)



