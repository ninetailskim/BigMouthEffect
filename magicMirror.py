import cv2
import numpy as np
import argparse
import copy
import math
import paddlehub as hub

class detUtils():
    def __init__(self):
        super(detUtils, self).__init__()
        self.lastres = None
        self.module = hub.Module(name="ultra_light_fast_generic_face_detector_1mb_320")
    
    def distance(self, a, b):
        return math.sqrt(math.pow(a[0]-b[0], 2) + math.pow(a[1]-b[1], 2))

    def iou(self, bbox1, bbox2):

        b1left = bbox1['left']
        b1right = bbox1['right']
        b1top = bbox1['top']
        b1bottom = bbox1['bottom']

        b2left = bbox2['left']
        b2right = bbox2['right']
        b2top = bbox2['top']
        b2bottom = bbox2['bottom']

        area1 = (b1bottom - b1top) * (b1right - b1left)
        area2 = (b2bottom - b2top) * (b2right - b2left)

        w = min(b1right, b2right) - max(b1left, b2left)
        h = min(b1bottom, b2bottom) - max(b1top, b2top)

        dis = self.distance([(b1left+b1right)/2, (b1bottom+b1top)/2],[(b2left+b2right)/2, (b2bottom+b2top)/2])

        if w <= 0 or h <= 0:
            return 0, dis
        
        iou = w * h / (area1 + area2 - w * h)
        return iou, dis


    def dodet(self, frame):
        result = self.module.face_detection(images=[frame], use_gpu=False)
        result = result[0]['data']
        if isinstance(result, list):
            if len(result) == 0:
                return None, None
            if len(result) > 1:
                if self.lastres is not None:
                    maxiou = -float('inf')
                    maxi = 0
                    mind = float('inf')
                    mini = 0
                    for index in range(len(result)):
                        tiou, td = self.iou(self.lastres, result[index])
                        if tiou > maxiou:
                            maxi = index
                            maxiou = tiou
                        if td < mind:
                            mind = td
                            mini = index  
                    if tiou == 0:
                        return result[mini], result
                    else:
                        return result[maxi], result
                else:
                    self.lastres = result[0]
                    return result[0], result
            else:
                self.lastres = result[0]
                return result[0], result
        else:
            return None, None

class LandmarkUtils():
    def __init__(self):
        super().__init__()
        self.module = hub.Module(name="face_landmark_localization")

    def predict(self, frame):
        result = self.module.keypoint_detection(images=[frame], use_gpu=False)
        if result is not None:
            if len(result) > 0:
                return result[0]['data']
            else:
                return None
        else:
            return None

def mirror(frame, cx=None, cy=None, radius=50):
    
    h, w = frame.shape[:2]
    if cx is None and cy is None:
        cx = w / 2
        cy = h / 2
    print(cx, cy)
    print(radius)
    nx = 0
    ny = 0
    radius = int(radius)
    backFrame = copy.deepcopy(frame)
    for i in range(w):
        for j in range(h):
            tx = i - cx
            ty = j - cy

            distance = tx **2 + ty ** 2
            if distance < radius ** 2:
                nx = int(tx * distance ** 0.5 / radius)
                ny = int(ty * distance ** 0.5 / radius)
                nx = int(nx + cx)
                ny = int(ny + cy)
                if ny < h and nx < w and ny > 0 and nx > 0:
                    backFrame[j, i, :] = frame[ny, nx, :]
    return backFrame


def magicMirrorByDet(args):
    DU = detUtils()
    cap = cv2.VideoCapture(0 if args.source == None else args.source)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.save, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if args.source == 0:
                frame = cv2.flip(frame, 1)
            bbox, _ = DU.dodet(frame)
            cx = None
            cy = None
            if bbox is not None:
                top = bbox['top']
                right = bbox['right']
                left = bbox['left']
                bottom = bbox['bottom']
                
                cx , cy = int((left + right) / 2) , int((top + bottom) / 2)
                ttimg = cv2.rectangle(frame, (int(left), int(top)),(int(right), int(bottom)),(0,0,255),5)
                cv2.imshow("ttimg", ttimg)
                cv2.waitKey(1)
            DFrame = mirror(frame, cx ,cy)
            if args.show:
                cv2.imshow("Debug", DFrame)
                cv2.waitKey(1)
            out.write(DFrame)
        else:
            break
    cap.release()
    out.release()   
    cv2.destroyAllWindows()

def magicMirrorByKP(args):
    DU = detUtils()
    LU = LandmarkUtils()
    cap = cv2.VideoCapture(0 if args.source == None else args.source)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.save, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if args.source == 0:
                frame = cv2.flip(frame, 1)
            res = LU.predict(frame)
            cx = None
            cy = None
            radius = 50
            if res is not None:
                res = res[0]
                cx = (res[51][0] + res[57][0] + res[62][0] + res[66][0]) / 4
                cy = (res[51][1] + res[57][1] + res[62][1] + res[66][1]) / 4
                radius = ((res[51][0] - res[57][0]) ** 2 + (res[51][1] - res[57][1]) ** 2) ** 0.5 * 1.5
                frame = mirror(frame, cx, cy, radius)
            if args.show:
                cv2.imshow("Debug", frame)
                cv2.waitKey(1)
            out.write(frame)
        else:
            break
    cap.release()
    out.release()   
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=None)
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--show', type=str, default=None)
    args = parser.parse_args()
    magicMirrorByKP(args)






