"""
    Runs YOLOv5 on a given video or camera in realtime
    and displays the results in a window.
    
    sample command to run the script
    
    python run_yolo.py --show --video_path test.mp4
    
    Requires:
        - cvu
        - numpy
        - opencv-python
        - tensorrt
        - CUDA 11.6
        - cudnn
        - pycuda
    
"""




from random import randint
import socket
import argparse
import numpy
import pickle   
from cvu.detector.yolov5 import Yolov5 as yolort
import cv2
import numpy as np
import json
import time 


Object_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]
COLORS = np.random.uniform(0, 255, size=(len(Object_classes), 3))
classes_to_colors = {Object_classes[i]: COLORS[i] for i in range(len(Object_classes))} 


def draw_bbox(image: np.ndarray,
              bbox: np.ndarray,
              className = None,
              thickness = 3) -> dict:
    """
    Draws a bounding box on the image inplace in the format of [x1, y1, x2, y2]
    
    """
    # generate random color every time you call draw_bbox
    color = classes_to_colors[className]
    
    if className == "UO":
        return 

    # convert cordinates to int
    x1, y1, x2, y2 = map(int, bbox[:4])

    # add title
    # if className == "person" or className == "car":
    #scale = min(image.shape[0], image.shape[1]) / (720 / 0.9)
    scale = 0.5
    text_size = cv2.getTextSize(className, 0, fontScale=0.5, thickness=1)[0]
    top_left = (x1 - thickness + 1, y1 - text_size[1] - 10)
    bottom_right = (x1 + text_size[0] + 5, y1)

    cv2.rectangle(image, top_left, bottom_right, color=color, thickness=-1)
    cv2.putText(image, className, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 2)
    # add box
    cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=thickness)

    



def put_fps(image: np.ndarray, fps: float) -> None:
    """
    Puts the FPS on the image inplace.
    """
    textsize = cv2.getTextSize(f"{fps:.2f} FPS", 0, fontScale=0.5, thickness=1)[0]
    cv2.putText(image, f"{fps:.2f} FPS", (image.shape[1] - textsize[0] - 10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    



if __name__ == '__main__':
    
    show_image = False
    
    
    parse = argparse.ArgumentParser()
    # show_image flag with default value False
    parse.add_argument('--show', default=False, action='store_true')
    parse.add_argument('--type', default='video', type=str)
    parse.add_argument('--image_path', default='test.jpg', type=str)
    # video path
    parse.add_argument('--video_path', default='camera')
    parse.add_argument('--show_fps', default=False, action='store_true')
    args = parse.parse_args()
    show_image = args.show
    video_path = args.video_path
    show_fps = args.show_fps
    if video_path == 'camera':
        video_path = 0
    


    image_width = 256
    image_height = 256
    
    
    
    if args.type == 'video':
        cam_reader = cv2.VideoCapture(video_path)
    elif args.type == 'image':
        img = cv2.imread(args.image_path)
    
    
    
    
    # should take some time to compile. With tensorrt, it is much faster to load the model
    # and it takes less memory
    model = yolort(auto_install=True, backend='tensorrt', weight="yolov5s", dtype='fp32', classes='coco')

    start_time = time.time()
    fc = 0
    display_time = 2
    fps = 0
    
    try:
        if args.type == 'video':
            while cam_reader.isOpened():
                
                
                # fps = cam_reader.get(cv2.CAP_PROP_FPS)
                
                
                
                ret, img = cam_reader.read()
                img = cv2.resize(img, (image_width, image_height))
                fc += 1
                TIME = time.time() - start_time
                
                if(TIME > display_time):
                    fps = fc / TIME
                    fc = 0
                    start_time = time.time()
                
                if ret:
                    preds = model(img)
                    for pred in preds:
                        draw_bbox(img, pred.bbox, pred.class_name, thickness=1)
                    if show_fps:
                        put_fps(img, float(fps))

                    #put_fps(img, fps)
                    if show_image:
                        cv2.imshow("Image", img)
                        ch=cv2.waitKey(1)
                        if ch ==27 or ch==ord('q') or ch==ord('Q'):
                            cv2.destroyAllWindows()
                            break
                
                    
        elif args.type == 'image':
            #img = cv2.resize(img, (image_width, image_height))
            preds = model(img)
            for pred in preds:
                draw_bbox(img, pred.bbox, pred.class_name, thickness=2)
            if show_image:
                cv2.imshow("Image", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            

    except KeyboardInterrupt:
        cam_reader.release()
        cv2.destroyAllWindows()
        print("KeyboardInterrupt")