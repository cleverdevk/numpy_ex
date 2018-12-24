import cv2
import numpy as np

print("start!")

img_training = []
img_evalutaion = []


for i in range(0,50000):
    img_R = []
    img_G = []
    img_B = []
    img_Result = []
    img = cv2.imread("./traning/traning_set_"+str(i)+".jpg")
    for j in range(0, 32):
        for k in range(0,32):
            img_R.append(img[j][k][0])
            img_G.append(img[j][k][1])
            img_B.append(img[j][k][2])
    img_Result = img_R + img_G + img_B
    img_training.append(img_Result)

    if i % 500 == 0:
        print("[IMG_TRAINING] "+ str(i) +"th item completed!")

np_img_training = np.array(img_training)
print("[IMG_TRAINING] TRAINING LENGTH : " + str(len(np_img_training)))
print("[IMG_TRAINING] Complete!")


for i in range(0, 10000):
    img_R = []
    img_G = []
    img_B = []
    img_Result = []
    img = cv2.imread("./traning/traning_set_" + str(i) + ".jpg")
    for j in range(0, 32):
        for k in range(0, 32):
            img_R.append(img[j][k][0])
            img_G.append(img[j][k][1])
            img_B.append(img[j][k][2])
    img_Result = img_R + img_G + img_B
    img_evalutaion.append(img_Result)

    if i % 500 == 0:
        print("[IMG_EVALUATION] "+ str(i) +"th item completed!")

np_img_evaluation = np.array(img_evalutaion)
print("[IMG_EVALUTATION] EVALUATION LENGTH : " + str(len(np_img_evaluation)))
print("[IMG_EVALUATION] Complete!")

print("[Numpy] Save Text './data/training.csv'")
np.savetxt("./data/training.csv", img_training, delimiter=",")
print("[Numpy] Save Text './data/test.csv'")
np.savetxt("./data/test.csv",img_evalutaion,delimiter=",")


print("done!")