
import time
import cv2
import numpy as np

IMAGE_PATH = "socolakeomut.jpg"

def scale_to_0_255(img):
    print(np.min(img))
    print(np.max(img))

    min_val = np.min(img)
    max_val = np.max(img)
    new_img = (img - min_val) / (max_val - min_val) # 0-1
    new_img *= 255
    return new_img

def apply_sliding_window_on_3_channels(img, kernel):
    # https://docs.opencv.org/4.4.0/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04
    layer_blue = cv2.filter2D(src=img[:,:,0], ddepth=-1, kernel=kernel)
    layer_green = cv2.filter2D(src=img[:,:,1], ddepth=-1, kernel=kernel)
    layer_red = cv2.filter2D(src=img[:,:,2], ddepth=-1, kernel=kernel)    
    
    new_img = np.zeros(list(layer_blue.shape) + [3])
    new_img[:,:,0], new_img[:,:,1], new_img[:,:,2] = layer_blue, layer_green, layer_red
    return new_img

def generate_gabor_bank(num_kernels, ksize=(15, 15), sigma=3, lambd=6, gamma=0.25, psi=0):
    bank = []
    theta = 0
    step = np.pi / num_kernels
    for idx in range(num_kernels):
        theta = idx * step
        # https://docs.opencv.org/4.4.0/d4/d86/group__imgproc__filter.html#gae84c92d248183bd92fa713ce51cc3599
        kernel = cv2.getGaborKernel(ksize=ksize, sigma=sigma, theta=theta, lambd=lambd, gamma=gamma, psi=psi)
        bank.append(kernel)
    return bank

def main():
    img = cv2.imread(IMAGE_PATH)
    gabor_bank = generate_gabor_bank(num_kernels=4)
    
    h, w, c = img.shape
    final_out = np.zeros([h, w*(len(gabor_bank)+1), c])
    final_out[:, :w, :] = img
    
    avg_out = np.zeros(img.shape)
    
    for idx, kernel in enumerate(gabor_bank):
        res = apply_sliding_window_on_3_channels(img, kernel)
        final_out[:, (idx+1)*w:(idx+2)*w, :] = res
        kh, kw = kernel.shape[:2]
        kernel_vis = scale_to_0_255(kernel)
        # https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html
        # https://numpy.org/doc/stable/reference/generated/numpy.repeat.html
        final_out[:kh, (idx+1)*w:(idx+1)*w+kw, :] = np.repeat(np.expand_dims(kernel_vis, axis=2), repeats=3, axis=2)        
        avg_out += res
        
    avg_out = avg_out / len(gabor_bank)
    avg_out = avg_out.astype(np.uint8)
    cv2.imwrite("minh_gabor.jpg", final_out)
    cv2.imwrite("minh_avg.jpg", avg_out)
    print("Saved @ minh_gabor.jpg")
    print("Saved @ minh_avg.jpg")
               
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    elapsed_time = end - start
    print('Elapsed time: %.2f second' % elapsed_time)
    print('---------')
    print('* Follow me @ ' + "\x1b[1;%dm" % (34) + ' https://www.facebook.com/kyznano/' + "\x1b[0m")
    print('* Minh fanpage @ ' + "\x1b[1;%dm" % (34) + ' https://www.facebook.com/minhng.info/' + "\x1b[0m")    
    print('* Join GVGroup @ ' + "\x1b[1;%dm" % (34) + 'https://www.facebook.com/groups/ip.gvgroup/' + "\x1b[0m")    
    print('* Thank you ^^~')   