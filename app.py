from asyncore import write
import calendar
import colorsys
import csv
import json
import math
import random
from flask import Flask, request, jsonify
from matplotlib import image
import skimage.io
import mrcnn.model as modellib
from mrcnn.config import Config
import cv2
import numpy as np
from collections import Counter
import time

app = Flask(__name__)
class_names = ['BG', 'FriedMusselPancakes', 'KhaoMokGai', 'PadPakBung', 'PadThai', 'Somtam', 'KaoManGai', 'PhatKaphrao', 'fried egg', 'KaiJeow', 'LarbMoo', 'rice']
class_names_th = ['BG', 'หอยทอด', 'ข้าวหมกไก่', 'ผัดผักบุ้ง', 'ผัดไทย', 'ส้มตำ', 'ข้าวมันไก่', 'ผัดกะเพรา', 'ไข่ดาว', 'ไข่เจียว', 'ลาบหมู', 'ข้าว']
p = {"PhatKaphrao" : [-0.1410, 0.4763, 0.2634, 0.1493, -124.9147, 192, 18, 3, 11],
    "KaoManGai" : [0.0068, 2.9861, -0.2278, 0.1223, -569.2506, 540, 22, 87, 12],
    "rice" : [0.5162, 4.1376, -3.5225, 0.1212, -295.2575, 300, 5, 66, 1.5]}

def estimateCalory(name, max, mean, min, len):
    if name in p:
        v = p[name]
        cal = (max * v[0]) + (mean * v[1]) + (min * v[2]) + (len * v[3]) + v[4]
        pro = (cal * v[6]) / v[5]
        carb = (cal * v[7]) / v[5]
        fat = (cal * v[8]) / v[5]
        return cal, pro, carb, fat
    else:
        return 0, 0, 0, 0

class foodConfig(Config):
    NAME = "food"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 11
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    BACKBONE = "resnet101"
    MEAN_PIXEL = np.array([158.73, 140.05, 106.13])
    MAX_GT_INSTANCES = 100
    DETECTION_MAX_INSTANCES = 100
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

config = foodConfig()

class InferenceConfig(config.__class__):
    # DETECTION_MIN_CONFIDENCE = 0.9
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config, model_dir="./")
model.load_weights('mask_rcnn_food_0120.h5', by_name=True)
model.keras_model._make_predict_function()

@app.route('/api/predict/test', methods=['GET', 'POST'])
def test():
    if request.files.get('image'):
        res = []
        calory, pro, carb, fat = estimateCalory(class_names[7], 280, 270, 1400)
        res.append({"name": class_names_th[7], "calory": "{:.2f}".format(calory), "protein": "{:.2f}".format(pro), 
            "carb": "{:.2f}".format(carb), "fat": "{:.2f}".format(fat)})
        return jsonify({"predict": res})
    return 'No image'

@app.route('/api/predict/', methods=['GET', 'POST'])
def predict():
    if request.files.get('image'):
        image = skimage.io.imread(request.files['image'])
        # skimage.io.imsave(f'read.jpg', image)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA) 
        # skimage.io.imsave(f're.jpg', image)
        results = model.detect([image], verbose=1)
        r = results[0]
        res = []
        for j in r['class_ids']:
            res.append(class_names[j])
            print(class_names[j])
        return jsonify(res)
    return "No Image"

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def extractDepth(x):
    depthConfidence = (x >> 13) & 0x7 
    if (depthConfidence > 6): return 0 
    return x & 0x1FFF 

def getDepthPercentage(x):
    depthConfidence = (x >> 13) & 0x7 
    if (depthConfidence == 0):
        return 1
    else: 
        return (depthConfidence - 1) / 7

def getDepthRange(x):
    return x & 0x1FFF 

@app.route('/api/depth/', methods=['GET', 'POST'])
def predict_depth():
    if request.files.get('image'):
        gmt = time.gmtime()
        ts = calendar.timegm(gmt)
        image = skimage.io.imread(request.files['image'])
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        skimage.io.imsave(f'pic/{ts}.jpg', image)
        
        logs = open(f'logs.txt', "a")
        logs.write("---------------------------------------------" + "\n")
        logs.write("file: " + str(ts) + "\n")

        results = model.detect([image], verbose=1)

        r = results[0]
        class_ids = r['class_ids'].tolist()

        res = []

        if (len(class_ids) == 0):
            print(res)
            return jsonify({"predict": res})
        
        f = open(f'depth2.csv', 'a', newline='')
        writer = csv.writer(f)

        depth = json.loads(request.form['depth'])

        depthArray = np.array(depth, dtype=np.uint16)

        # depthMap = np.array([extractDepth(x) for x in depthArray]).reshape(90, 160)
        # depthMap = [getDepthPercentage(x) for x in depthArray]
        # skimage.io.imsave(f'pic/{ts}_d.jpg', depthMap)
        # depthMap_1D = np.hstack(depthMap)
        # print("depthCon:", (depthMap == 0).sum())

        depthRange = np.array([getDepthRange(x) for x in depthArray]).reshape(90, 160)
        skimage.io.imsave(f'pic/{ts}_d.jpg', depthRange)
        depthRange_1D = np.hstack(depthRange)

        image = cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA)
        masks = np.uint8(r['masks'])
        color = random_colors(len(class_ids))

        # img_d = skimage.io.imread(f'pic/{ts}_d.jpg')

        # index = np.argwhere(depthRange > mostDepth[0][0])
        # mask = np.zeros(img_d.shape)
        # for x, y in index:
        #     mask[x][y] = 1
        # colors = random_colors(2)

        # img_d = cv2.merge([img_d, img_d, img_d])
        # gray = apply_mask(img_d, mask, colors[1])
        # skimage.io.imsave(f'pic/{ts}_dm.jpg', gray)

        # img_d = skimage.io.imread(f'pic/{ts}_dr.jpg')
        # img_d = cv2.merge([img_d, img_d, img_d])

        for i in range(len(class_ids)):
            d = []

            sub_mask = cv2.resize(masks[:, :, i], (160, 90), interpolation=cv2.INTER_AREA)
            # mask_r[:, :, i] = sub_mask
            index = np.argwhere(sub_mask == True)

            img = apply_mask(image, sub_mask, color[i])
            skimage.io.imsave(f'pic/{ts}_mask{i}.jpg', img)

            for x, y in index:
                d.append(int(depthRange[x][y]))
            
            w = open(f'depth/{ts}.txt', 'w')
            w.write(json.dumps(d))
            w.close()

            mean_mask = np.mean(d)
            max_mask = max(d)
            min_mask = min(d)
            print("max in mask:", max_mask, "min in mask:", min_mask, "mean in mask", mean_mask)
            logs.write("max in mask: " + str(max_mask) + " ,min in mask: " + str(min_mask) + "\n" + " ,mean in mask: " + str(mean_mask) + "\n")

            calory, pro, carb, fat = estimateCalory(name=class_names[class_ids[i]], max=max_mask, mean=mean_mask, len=len(d), min=min_mask)

            res.append({"name": class_names_th[class_ids[i]], "calory": "{:.2f}".format(calory), "protein": "{:.2f}".format(pro), 
            "carb": "{:.2f}".format(carb), "fat": "{:.2f}".format(fat)})
            logs.write("name: " + class_names[class_ids[i]] + " ,calory: " + "{:.2f}".format(calory) + "\n")
            writer.writerow([max_mask, min_mask, mean_mask, len(d), class_names[class_ids[i]], ts])
        
        logs.write("---------------------------------------------" + "\n")
        logs.close()

        f.close()

        print(res)
        return jsonify({"predict": res})
    return "No Image"

@app.route('/', methods=['GET'])
def home():
    return "Bnn-project ml API"

if __name__ == '__main__':
    app.run()