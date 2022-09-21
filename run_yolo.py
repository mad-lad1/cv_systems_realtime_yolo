import gi
gi.require_version('Gtk', '2.0')

import jsonpickle
import socket
import argparse
import numpy
import pickle
from cvu.detector.yolov5 import Yolov5 as yolort
import cv2
import numpy as np
import GLOBALS
from myUtils.ioUtils import RGBCAM
import json
from Product.Tracking.detection import transform_6points_to_bb_box, Detection
from Product import IntrinsicCalibratorContainer
### THIS CODE IS INCOMPLETE
## VIRTUAL CURRUS UNIT SHOULD BE CREATED AND USED TO CREATE HOMOGRAPHY
## DELETE THIS TEXT ONCE DONE

Object_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                          'traffic light',
                          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                          'cow',
                          'elephant', 'bear', 'zebra', 'giraffe'] + 56 * ['UO']

APP_PORT= GLOBALS.EXTERNAL_COM_PORT
INTERNAL_COM_PORT = GLOBALS.INTER_COM_PORT
INTERNAL_COM_IP=GLOBALS.INTER_COM_URL

def load_intrinsic_params(camera_name) -> dict:
    """Loads the intrinsic parameters of the camera
    :param camera_name: the name of the camera
    :return: the intrinsic parameters of the camera
    """
    path = "./Product/Data/Configs/Intrinsic/%s/CalibrationMatrix.pkl" % camera_name
    with open(path, 'rb') as f:
        return pickle.load(f)


def sendPacket(messageText:str):
    serverAddressPort = (INTERNAL_COM_IP, INTERNAL_COM_PORT)
    UDPClientSocket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    UDPClientSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    UDPClientSocket.sendto(messageText.encode('utf-8'), serverAddressPort)


#draws the bounding box and the class name of the detected object in the image
def draw_bbox(image: np.ndarray,
              bbox: np.ndarray,
              className = None,
              thickness = 3) -> dict:
    """Draw Bounding Box on the given image (inplace
    """
    # generate random color
    if className == "person":
        #green color
        color = (0, 255, 0)
    elif className == "car":
        #red color
        color = (0, 0, 255)

    # convert cordinates to int
    x1, y1, x2, y2 = map(int, bbox[:4])

    # add title
    if className == "person" or className == "car":
        scale = min(image.shape[0], image.shape[1]) / (720 / 0.9)
        text_size = cv2.getTextSize(className, 0, fontScale=scale, thickness=1)[0]
        top_left = (x1 - thickness + 1, y1 - text_size[1] - 20)
        bottom_right = (x1 + text_size[0] + 5, y1)

        cv2.rectangle(image, top_left, bottom_right, color=color, thickness=-1)
        cv2.putText(image, className, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), 2)
        # add box
        cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=thickness)

    #make homography projection
    #dets,dets_bev=virtualCurrusUnit.process_detections(dets, H, img_w, img_h)



def project_to_bev(H, bbox):
    minx, miny, maxx, maxy = bbox[:4]
    cent_x, cent_y = (minx + maxx) / 2, (miny + maxy) / 2
    xs = [minx, maxx, cent_x]
    ys = [miny, maxy, cent_y]
    coords = np.array([xs, ys, [1,] * len(xs)])
    bev_coordinates = np.matmul(H, coords)
    bev_coordinates = bev_coordinates / bev_coordinates[-1:, :]
    projected_bbox = transform_6points_to_bb_box(bev_coordinates)

    return projected_bbox

def prepare_detection(bbox, className: str, confidence) -> Detection:
    det = Detection(
        tlwh= np.asarray(bbox, dtype=np.float),
        confidence=confidence,
        feature=None,
        crop=None,
        label=className,
    )
    return det

def prepare_packet(dets, dets_bev: list, cam_ip: str):
    packet = {
        "dets": dets,
        "dets_bev": dets_bev,
        "cam_ip": cam_ip
    }
    return packet


def load_homography(system_name: str, camera_name: str) -> np.ndarray:
    """Loads the homography matrix of the system
    :param system_name: the name of the system
    :return: the homography matrix of the system
    """
    path = "./Product/Data/Systems/%s/Sensors/%s/Extrinsic/homography_%s.pkl" % (system_name, camera_name, camera_name)
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return np.eye(3)


if __name__ == '__main__':
    #provide sys.argv[1] as the name of the system and sys.argv[2] as the name of the camera


    p = argparse.ArgumentParser()
    p.add_argument('-s', '--system', help='system name', required=True)
    p.add_argument('-c', '--camera', help='camera name', required=True)
    p.add_argument('-sh', '--show', help='show image', action='store_true')

    args = p.parse_args()

    #get show image flag
    show_image = args.show
    #get system name
    system_name = args.system
    #get camera name
    camera_name = args.camera

    print("System: %s" % system_name)
    print("Camera: %s" % camera_name)
    print("Show Image: %s" % show_image)

    image_width = 640
    image_height = 640
    url = "Stream"
    cam_reader = cv2.VideoCapture(0)
    model = yolort(auto_install=True, backend='tensorrt', weight="yolov5s", dtype='fp16', classes=Object_classes)

    UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    #to test

    H = load_homography(system_name, camera_name)

    packet = {}


    intrinsic_params = load_intrinsic_params("MICROSOFTCAM")
    distortion_coeffs = intrinsic_params['dist']
    mtx = intrinsic_params['mtx']
    try:

        while True:
            dets_bev = []
            dets = []
            ret, img = cam_reader.read()
            img = cv2.resize(img, (image_width, image_height))
            img_undistorted = IntrinsicCalibratorContainer.IntrinsicCalibrationUnit.undistortImageStatic(img, distortion_coeffs, mtx)
            if ret:
                preds = model(img_undistorted)
                for pred in preds:
                    draw_bbox(img_undistorted, pred.bbox, pred.class_name)
                    if pred.class_name == "person" or pred.class_name == "car":
                        det = prepare_detection(pred.bbox, pred.class_name, pred.confidence)
                        projected_bbox = project_to_bev(H, pred.bbox)
                        det_bev = prepare_detection(projected_bbox, pred.class_name, pred.confidence)
                        dets_bev.append(det_bev)
                        dets.append(det)
                        print(det_bev.tlwh)


                packet = prepare_packet(dets, dets_bev, url)
                #pickle the packet
                pickled_packet = jsonpickle.dumps(packet)

                pickled_packet = json.dumps(pickled_packet)
                sendPacket(pickled_packet)

                if show_image:
                    cv2.imshow("Image", img_undistorted)
                ch=cv2.waitKey(1)
                if ch ==27 or ch==ord('q') or ch==ord('Q'):
                    cv2.destroyAllWindows()
                    UDPClientSocket.close()
                    break

    except KeyboardInterrupt:
        UDPClientSocket.close()
        cv2.destroyAllWindows()
        print("KeyboardInterrupt")
            
# if __name__ == '__main__':
    
#     im = cv2.imread('.././Cats.jpg')
#     url = "rtsp://admin:curruS101@192.168.0.135/2"
#    # video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)

#     # im = im.astype(float)/255.0
#     video_capture = cv2.VideoCapture(1)
#     print(np.max(im), np.min(im))
#     # im = np.stack((im,im),axis=0)
#     model = yolort(auto_install=True,backend='tensorrt', weight="yolov5s",dtype='fp16',classes=Object_classes)
#     while True:
        
#         ret, im = video_capture.read()
#         im = cv2.resize(im, (640, 640))
#         preds =model(im.copy())
#         for pred in preds:
#             print("postion", pred.bbox)
#         im2 = im.copy()
#         im_anot = preds.draw(im2)
#         cv2.imshow('a',im_anot)
#         #cv2.imshow('a', im2)
#         cv2.waitKey(1)