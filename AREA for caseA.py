import torch
import numpy as np
import cv2
from time import time, strftime, localtime
from datetime import datetime

import pandas as pd

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import matplotlib
import matplotlib.pyplot as plt
 
# Restricted Area
#좌측 상단
x_1, y_1 = 600, 400
#우측 상단
x_2, y_2 = 1000, 400
#우측 하단
x_3, y_3 = 1200, 520
#좌측 하단
x_4, y_4 = 500, 520

# 중심좌표
x_b_center = int(((x_1+x_2)/2+(x_3+x_4)/2)/2)
y_b_center = int((y_1+y_3)/2)

pts1 = np.array([[x_1,y_1], [x_2,y_2], [x_3,y_3],[x_4,y_4]], dtype=np.int32) # 사용자 눈에 보이는 polygon 생성을 위한 좌표 지정
polyList = [(x_1,y_1),(x_2,y_2),(x_3,y_3),(x_4,y_4)] # 기계 연산에 사용할 polygon 생성을 위한 좌표 지정
polygon = Polygon(polyList) # Polygon 함수를 이용해 제한 구역 지정

class ObjectDetection:
    def __init__(self, url, out_file):
        # 객체 생성 시 호출
        self._URL = url
        self.model = self.load_model()
        self.classes = self.model.names
        self.out_file = out_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_video_from_url(self):
        return cv2.VideoCapture('/home/yun/YUNJUNGCHAN/data/site_A_2personals.mp4')
        #return cv2.VideoCapture(0)

    def load_model(self):
        # YOLOv5 모델 로드
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        # frame: 단일 프레임; numpy/list/tuple 형식
        # return: 프레임에서 모델이 감지한 객체의 레이블과 좌표
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
        return labels, cord

    def class_to_label(self, x):
        # x 숫자 레이블 -> 문자열 레이블로 반환
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        # 경계상자와 레이블을 프레임에 플로팅
        # results: 프레임에서 모델이 감지한 객체의 레이블과 좌표
        # frame: 점수화된 프레임
        # return: 경계 상자와 레이블이 플로팅된 프레임
        
        y_val_temp = 0
        
        blue = (255,0,0)
        green = (0,255,0)
        red = (0,0,255)
        black = (0,0,0)
        green2 = (0,102,51)
        plum = [153,0,51]
        white = (255,255,255)

        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        
        x1_val = []
        y1_val = []
        x2_val = []
        y2_val = []
        
        for i in range(n):
            row = cord[i]
            
            if self.class_to_label(labels[i]) == 'person':
                if row[4] >= 0.01:
                    x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                    

                    xc = int((x2+x1)/2)
                    yc = int(y2-20)
                    bgr = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                    cv2.putText(frame, self.class_to_label(labels[i])
                                + ': ' + str(x1) + ', ' + str(x2) + ', ' + str(y1) + ', ' + str(y2),
                                (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                    cv2.circle(frame, (xc, yc), 10, (0,0,255), 5)
                    result = polygon.contains(Point(xc,yc))
                    if result == True :
                        #print('작업자가 B구역에 들어와 있습니다.')
                        y_val_temp += 1
                        x1_val.append(x1)
                        y1_val.append(y1)
                        x2_val.append(x2)
                        y2_val.append(y2)
                    
            if self.class_to_label(labels[i]) == 'truck':
                if row[4] >= 0.2:
                    x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), blue, 2)
                    cv2.putText(frame, self.class_to_label(labels[i])
                                + ': ' + str(x1) + ', ' + str(x2) + ', ' + str(y1) + ', ' + str(y2),
                                (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, blue, 2)
        cv2.polylines(frame, [pts1], True, red, thickness=2) # 사용자 눈에 보이는 polygon 그리기
        #cv2.putText(frame, "Restricted Area", (x_b_center-210, y_b_center+15), cv2.FONT_HERSHEY_PLAIN, 3, red, 2, cv2.LINE_AA)
        cv2.putText(frame, "RA", (x_b_center-75, y_b_center+35), cv2.FONT_HERSHEY_PLAIN, 7, red, 7, cv2.LINE_AA)
        
        Productivity = False
        x1_min = 0
        y1_min = 0
        x2_max = 0
        y2_max = 0
        
        if len(x1_val) >= 2 : 
            x1_min = min(x1_val)
            y1_min = min(y1_val)
            x2_max = max(x2_val)
            y2_max = max(y2_val)
            if y_val_temp >= 2 :
                Productivity = True
                cv2.rectangle(frame, (x1_min-20, y1_min-20), (x2_max+20, y2_max+20), plum, 2)
        else :
            Productivity = False
                 
        return frame, y_val_temp, x1_min, y1_min, Productivity, x1_val

    def __call__(self):
        matplotlib.use('TkAgg')
        # 인스턴스 생성 시 호출; 프레임 단위로 비디오 로드
        player = self.get_video_from_url()
        
        blue = (255,0,0)
        green = (0,255,0)
        red = (0,0,255)
        black = (0,0,0)
        green2 = (0,102,51)
        plum = [153,0,51]
        white = (255,255,255)
        
        x_val_temp = 0
        time_val_temp = 0
        count = 0
        count_c = 0
        count_a = 0
        
        x_val = []
        y_val = []
        y_val_c = []
        y_val_a = []
        
        time_val = []
        
        count_val = []
        count_val_c = []
        count_val_a = []
        
        fps_val = []
        
        before = 0
        
        while player.isOpened() :
            
            if not player:
                break
            x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
            y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
            four_cc = cv2.VideoWriter_fourcc(*"MJPG")
            out = cv2.VideoWriter(self.out_file, four_cc, 20, (x_shape, y_shape))
            #temp = 0
            while True:
                start_time = time()
                
                time_val.append(strftime('%Y-%m-%d %I:%M:%S %p', localtime()))
                x_val_temp += 1
                x_val.append(x_val_temp)
                
                
                ret, frame = player.read()
                assert ret
                results = self.score_frame(frame)
                frame_1, y_val_temp, x1_min, y1_min, Productivity, x1_val = self.plot_boxes(results, frame)
                frame = frame_1
                
                y_val.append(y_val_temp)
                     
                if len(y_val) >= 2 :
                    if y_val[-1] > y_val[-2] :
                        count += 1
                        
                count_val.append(count)
                 
                x_val_s = np.array(x_val)
                y_val_s = np.array(y_val)
                # fps_val_s = np.array(fps_val)
                
                if len(x1_val) >= 1 : 
                    if Productivity == True:
                        now = datetime.now()
                        #print((now-before).seconds)
                        if (now-before).seconds < 3:
                            cv2.rectangle(frame, (x1_min - 40, y1_min - 55), (x1_min + 150, y1_min - 25), black, -1)
                            cv2.putText(frame, "Suspecting.." ,(x1_min - 30, y1_min - 30 ), cv2.FONT_HERSHEY_SIMPLEX, 0.9, white, 2)
                            
                        if (now-before).seconds >= 3:
                            cv2.rectangle(frame, (x1_min - 45, y1_min - 55), (x1_min + 170, y1_min - 25), black, -1)
                            cv2.putText(frame, "Unsafe action" ,(x1_min - 40, y1_min - 30 ), cv2.FONT_HERSHEY_SIMPLEX, 0.9, white, 2)
                if Productivity == False:
                    before = datetime.now()
                                   
                out.write(frame)
                
                cv2.imshow('result', frame)
                if cv2.waitKey(1) == ord('q'):
                    break
        
                plt.subplot(1, 1, 1) 
                plt.title('The number of personels in Restricted Area')
                plt.plot(x_val_s, y_val_s, color='k',  label = 'Restricted Area')
                plt.ylabel('Person')
                #plt.legend()
                
                if x_val_temp >= 1:
                    plt.xlim(x_val_temp-100, x_val_temp+100)
                    plt.ylim(0,y_val_s[-1] + 2)

                while x_val_temp ==0:
                    break

                plt.pause(0.00001) 
                
                end_time = time()
                fps = 1/np.round(end_time - start_time, 3)
                #print(f"Frames Per Second : {fps}")
                
                fps_val.append(fps.round(1))
                
                dict = {'Time':time_val,'Frame':x_val, 'Person_inRA':y_val,'Count_inRA':count_val,'fps': fps_val} # RA : Restricted Area
                Result = pd.DataFrame(dict)
                Result.to_csv("Result.csv")
                
                print(Result[['Time', 'Person_inRA', 'Count_inRA', 'fps']].tail(5))
    plt.show()
                                    
Video = ObjectDetection("0", "Result.avi")
Video()

