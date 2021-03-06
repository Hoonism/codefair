from retinaface import RetinaFace
import matplotlib.pyplot as plt
import cv2
import math
import sys

import detectcopy

def run(always=False):
    for img in img_list:
        if always:
            return detectcopy.run(finding="cane", source=img, model="cane.pt")
        else:
            print(detectcopy.run(finding="wheelchair.pt", source="wheel chair"))
            ageProto="age_deploy.prototxt"
            ageModel="age_net.caffemodel"
            img_list = ["crosswalk_left.jpg", "crosswalk_right.jpg"]
            MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
            ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
            ageNet=cv2.dnn.readNet(ageModel,ageProto)
            faces = RetinaFace.extract_faces(img_path=img, align=True)
            print
            for face in faces:
                # plt.imshow(face)
                # plt.show()
                blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                ageNet.setInput(blob)
                agePreds=ageNet.forward()
                age=ageList[agePreds[0].argmax()]
                # print(f'Age: {age[1:-1]} years')
                if age == "(0-2)" or age == "(4-6)" or age == "(60-100)":
                    return 1
        
    return 0

def present():
    return 1
print(detectcopy.run(source="cargreen.png", ))