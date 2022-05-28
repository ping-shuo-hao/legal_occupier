import torchvision
from matplotlib import pyplot as plt
import cv2
import torchvision.transforms as T
from PIL import Image
import os
import tempfile
import warnings

warnings.filterwarnings('ignore')

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) 
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def get_prediction(img, threshold):
    """
    get_prediction
    parameters:
      - img_path - path of the input image
      - threshold - threshold value for prediction score
    method:
      - Image is obtained from the image path
      - the image is converted to image tensor using PyTorch's Transforms
      - image is passed through the model to get the predictions
      - class, box coordinates are obtained, but only prediction score > threshold
        are chosen.

    """
    # img = Image.open(img_path)
    #cv2.imread(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return pred_boxes, pred_class

# Return index of the closest bbox to the point
def closest_bbox(boxes, point, pred_cls, typ):
    closest = 99999
    index = -1
    for i in range(len(boxes)):
        bbox = boxes[i]
        clss = pred_cls[i]
        xAvg = (round(bbox[0][0]) + round(bbox[1][0]))/2
        #yAvg = (round(bbox[0][1]) + round(bbox[1][1]))/2
        xDiff = abs(xAvg - point)
        if (clss == typ or (clss == "car" and typ == "truck") or (clss == "truck" and typ == "car")) and (index < 0 or xDiff < closest):
            closest = xDiff
            index = i
    return index

# Return two images,
# the bbox in the last frame of 1st video, the bbox in the first frame of the 2nd video
def extract_bboxes(link1, link2, point1, point2, typ, threshold=0.5):
    # extract last frame in video at link1
    video = cv2.VideoCapture(link1)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if length > 1:
        video.set(1, length - 1)
    ret, frame = video.read()

    # get bboxes in last frame of video
    try:
        boxes, pred_cls = get_prediction(frame, threshold)
    except:
        return None, None
    
    # identify bbox closest to point1
    index = closest_bbox(boxes, point1, pred_cls, typ)
    if index >= 0:
        bbox = boxes[index]
        x1 = round(bbox[0][0])
        y1 = round(bbox[0][1])
        x2 = round(bbox[1][0])
        y2 = round(bbox[1][1])
        box_img1 = frame[y1:y2, x1:x2]
    else:
        return None, None

    # extract first frame in video at link2
    video = cv2.VideoCapture(link2)
    ret, frame = video.read()

    # get bboxes in first frame of video
    try:
        boxes, pred_cls = get_prediction(frame, threshold)
    except:
        return None, None
    
    # identify bbox closest to point2
    index = closest_bbox(boxes, point2, pred_cls, typ)
    if index >= 0:
        bbox = boxes[index]
        x1 = round(bbox[0][0])
        y1 = round(bbox[0][1])
        x2 = round(bbox[1][0])
        y2 = round(bbox[1][1])
        box_img2 = frame[y1:y2, x1:x2]
    else:
        return None, None

    return box_img1, box_img2