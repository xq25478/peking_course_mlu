from src import detect_faces, show_bboxes
from PIL import Image
import os

images = os.listdir("./images")
for i in range(1, 6):
    print("Detecting: ",images[i])
    image = Image.open("./images/" + images[i])

    image = image.resize((480, 480), Image.LANCZOS)

    bounding_boxes, landmarks = detect_faces(image, thresholds=[0.6, 0.7, 0.85])
    img = show_bboxes(image, bounding_boxes, landmarks)

    img.save("output/test_result_cpu_" + images[i])
