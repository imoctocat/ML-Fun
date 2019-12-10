import os
from os.path import exists, join, basename
import argparse
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True,
	help="path to folder with videos to be tested")
args = vars(ap.parse_args())

def download(name,cmd):
    """
    Executes the terminal command to download anything
    by Taking in its name & respective command to download
    name:str
    cmd:str
    """
    
    if not exists(name):
        os.system(cmd)
        print("{} download complete!".format(name))
    else:
        print("{} already exists!".format(name))


## Downloading the Git Repo
project_name = "deep_sort_pytorch"
git_cmd = 'git clone -q --recursive https://github.com/ZQPei/deep_sort_pytorch.git'
download(project_name,git_cmd)


import sys
sys.path.append(project_name)
sys.path.append(join(project_name, 'YOLOv3'))


## Downloading the pre-trained weights
wt_name = 'yolov3.weights'
wt_cmd = 'wget -q https://pjreddie.com/media/files/yolov3.weights'
download(wt_name,wt_cmd)  

ckpt_name = 'ckpt.t7'
ckpt_cmd = 'curl -Lb ./cookie "https://drive.google.com/uc?export=download&id=1_qwTWdzT9dWNudpusgKavj_4elGgbkUN" -o ckpt.t7'
download(ckpt_name,ckpt_cmd)  


## Loading the Models with pretrained weights

import cv2
import time

from YOLOv3 import YOLOv3
from deep_sort import DeepSort
from util import draw_bboxes
yolo3 = YOLOv3("deep_sort_pytorch/YOLOv3/cfg/yolo_v3.cfg","yolov3.weights","deep_sort_pytorch/YOLOv3/cfg/coco.names", is_xywh=True)
deepsort = DeepSort("ckpt.t7")



## Generating the outputs
tim1 = time.time()
names = os.listdir(args["folder"])
names.sort()
i = 0
for name in names:
    path_to_file = os.path.join(args["folder"],name)
    print("File to process : ",name)
    video_capture = cv2.VideoCapture()
    
    if video_capture.open(path_to_file):
      width, height = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
      fps = video_capture.get(cv2.CAP_PROP_FPS)
      i += 1
      video_writer = cv2.VideoWriter("outvid"+ str(i)+"--"+name.split('.')[0]+".avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
      continue
      #print("outvid"+ str(i)+"--"+name.split('.')[0]+".avi")
      while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
          break

        start = time.time()
        xmin, ymin, xmax, ymax = 0, 0, width, height
        im = frame[ymin:ymax, xmin:xmax, (2,1,0)]
        bbox_xywh, cls_conf, cls_ids = yolo3(im)
        if bbox_xywh is not None:
            mask = cls_ids==0
            bbox_xywh = bbox_xywh[mask]
            bbox_xywh[:,3] *= 1.2
            cls_conf = cls_conf[mask]
            outputs = deepsort.update(bbox_xywh, cls_conf, im)
            if len(outputs) > 0:
                bbox_xyxy = outputs[:,:4]
                identities = outputs[:,-1]
                frame = draw_bboxes(frame, bbox_xyxy, identities, offset=(xmin,ymin))

        end = time.time()
        print("time: {}s, fps: {}".format(end-start, 1/(end-start)))

        video_writer.write(frame)
      video_capture.release()
      video_writer.release()

      # .
    else:
      print("can't open the given input video file!",path_to_file)
tim2 = time.time()
print("That took {} mins time".format((tim2-tim1)/60))