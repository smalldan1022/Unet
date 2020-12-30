import cv2
import numpy as np
import matplotlib.pyplot as plt

# Plot the results

def plot_model_result(train_history):
    
    # Accuracy Loss figure
    plt.figure(0, figsize = (16,6))
    plt.subplot(121)
    
    plt.plot(train_history.history['accuracy'])
    plt.plot(train_history.history['val_accuracy'])
    plt.title('Train History')
    plt.legend(['train', 'validation'], loc = 'best')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.subplot(122)
    
    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.title('Train History')
    plt.legend(['train', 'validation'], loc = 'best')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('Accuracy_Loss.png')


    # IOU Loss figure
    plt.figure(1, figsize = (16,6))
    plt.subplot(121)
    
    plt.plot(train_history.history['mean_io_u'])
    plt.plot(train_history.history['val_mean_io_u'])
    plt.title('Train History')
    plt.legend(['train', 'validation'], loc = 'best')
    plt.xlabel('Epoch')
    plt.ylabel('IOU')
    
    plt.subplot(122)
    
    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.title('Train History')
    plt.legend(['train', 'validation'], loc = 'best')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('IOU_Loss.png')



# Fill the image holes

def Fill_hole(img):
    
    img_copy = img.copy()

    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)  

    ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)  

    contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  

    cv2.drawContours(img_copy,contours,-1,(255,255,255),-1)

    return img_copy


# Set the correct size of the buttom image and upper image , set the channel value of the upper image

def Set_channel_values(infer, origin, R, G, B):
    
    infer = infer / 255
    infer[infer>0.5] = 1
    infer[infer<=0.5] = 0

    (h, w, _) = infer.shape

    infer = cv2.resize(infer, (w//4, h//4))

    (h, w, _) = infer.shape

    origin = origin[0:h,0:w]

    print(origin.shape, infer.shape)


    infer[:,:,0][infer[:,:,0]==1] = B
    infer[:,:,1][infer[:,:,1]==1] = G
    infer[:,:,2][infer[:,:,2]==1] = R

    infer = np.array(infer, dtype=np.uint8)
    
    return infer, origin