import cv2
import numpy as np

# open video file
video_path = 'Nxde-Idle.mp4'
cap = cv2.VideoCapture(video_path)

output_size = (187, 333) # (width, height) 휴대폰 크기에 맞는 영상 사이즈

# initialize writing video
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('%s_output.mp4' % (video_path.split('.')[0]), fourcc, cap.get(cv2.CAP_PROP_FPS), output_size)
#인자, 파일저장이름/ 코덱/ FPS 1초당 프레임, cap.get 우리가 불러온 동영상과 똑같은 프레임으로 저장/ 저장 사이즈
# 영상의 크기는 고정이고 오브젝트 트래킹 이후 box의 크기는 유동적이어서 잘 계산해야한다.

# check file is opened : 동영상이 제대로 로드되면 True를 반환한다.
if not cap.isOpened():
    exit()

tracker = cv2.TrackerCSRT_create() # OPENCV_OBJECT_TRACKERS 중 적당한 csrt사용, 정확도가 높으면 속도가 느림

ret, img = cap.read() # 첫번째 프레임이 img에 저장

#ROI : Region of Interest, 관심영역
cv2.namedWindow('Select Window') #이 프로그램이 ROI에서 window를 사용하는구나 알 수 있게 이름 지정
cv2.imshow('Select Window', img) # Select Window의 첫번째 프레임을 보여줘라

# select ROI ROI를 설정하여 rect로 반환한다.
# 코드 실행 후 이미지에서 네모 박스로 영역지정 후 space바 누르면 ROI정보가 rect에 저장됨
rect = cv2.selectROI('Select Window', img, fromCenter=False, showCrosshair=True)
cv2.destroyWindow('Select Window')

# intialize tracker 오브젝트 트래커가 img와 rect를 따라가도록 설정(첫번째 프레임과 ROI 따라)
tracker.init(img,rect)



#이미지를 프레임마다 읽어서 ROI와 새로운 이미지랑 비교하여 따라가게 만든다.update함수 이용하여
while True:
    ret, img = cap.read()

    if not ret:
        exit()

    success, box= tracker.update(img)  # success : 성공했냐 안했냐/ box : rect형태의 데이터

    left, top, w, h = [int(v) for v in box] # box에 있는 데이터 가져오기

    # 영상의 크기는 고정이고 오브젝트 트래킹 이후 box의 크기는 유동적이어서 잘 계산해야한다.
    center_x = left + w / 2
    center_y = top +h / 2

    result_top = int(center_y - output_size[1] / 4)
    result_bottom = int(center_y + output_size[1] / 4 * 3)
    result_left = int(center_x - output_size[0] / 2)
    result_right = int(center_x + output_size[0] / 2)

    result_img = img[result_top:result_bottom, result_left:result_right].copy() # ROI지정 박스 안보이게

    out.write(result_img) #이미지를 동영상으로 저장한다.
    # result_img가 영상 밖으로 나가게 되면 에러가 난다.
    cv2.rectangle(img, pt1=(left,top), pt2=(left + w, top+h), color=(255,255,255), thickness=3)
    cv2.imshow('result_img', result_img)
    cv2.imshow('img',img)
    if cv2.waitKey(1) == ord('q'):
      break

    '''
    try:
        cv2.imshow('result_img', result_img)
        cv2.imshow('img',img)
        if cv2.waitKey(1) == ord('q'):
            break
    except:
    # select ROI ROI를 설정하여 rect로 반환한다.
    # 코드 실행 후 이미지에서 네모 박스로 영역지정 후 space바 누르면 ROI정보가 rect에 저장됨
        rect = cv2.selectROI('Select Window', img, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow('Select Window')
    '''
  

'''
import cv2
import numpy as np

# open video file
video_path = 'Nxde-Idle.mp4'
cap = cv2.VideoCapture(video_path)

output_size = (187, 333) # (width, height)
fit_to = 'height'

# initialize writing video
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('%s_output.mp4' % (video_path.split('.')[0]), fourcc, cap.get(cv2.CAP_PROP_FPS), output_size)

# check file is opened
if not cap.isOpened():
  exit()

# initialize tracker 
OPENCV_OBJECT_TRACKERS = {
  "csrt": cv2.TrackerCSRT_create,
  #"kcf": cv2.TrackerKCF_create,
  #"boosting": cv2.TrackerBoosting_create,
  #"mil": cv2.TrackerMIL_create,
  #"tld": cv2.TrackerTLD_create,
  #"medianflow": cv2.TrackerMedianFlow_create,
  #"mosse": cv2.TrackerMOSSE_create
}

tracker = OPENCV_OBJECT_TRACKERS['csrt']()

# global variables
top_bottom_list, left_right_list = [], []
count = 0

# main
ret, img = cap.read()

cv2.namedWindow('Select Window')
cv2.imshow('Select Window', img)

# select ROI
rect = cv2.selectROI('Select Window', img, fromCenter=False, showCrosshair=True)
cv2.destroyWindow('Select Window')

# initialize tracker
tracker.init(img, rect)

while True:
  count += 1
  # read frame from video
  ret, img = cap.read()

  if not ret:
    exit()

  # update tracker and get position from new frame
  success, box = tracker.update(img)
  # if success:
  left, top, w, h = [int(v) for v in box]
  right = left + w
  bottom = top + h

  # save sizes of image
  top_bottom_list.append(np.array([top, bottom]))
  left_right_list.append(np.array([left, right]))

  # use recent 10 elements for crop (window_size=10)
  if len(top_bottom_list) > 10:
    del top_bottom_list[0]
    del left_right_list[0]

  # compute moving average
  avg_height_range = np.mean(top_bottom_list, axis=0).astype(np.int)
  avg_width_range = np.mean(left_right_list, axis=0).astype(np.int)
  avg_center = np.array([np.mean(avg_width_range), np.mean(avg_height_range)]) # (x, y)

  # compute scaled width and height
  scale = 1.3
  avg_height = (avg_height_range[1] - avg_height_range[0]) * scale
  avg_width = (avg_width_range[1] - avg_width_range[0]) * scale

  # compute new scaled ROI
  avg_height_range = np.array([avg_center[1] - avg_height / 2, avg_center[1] + avg_height / 2])
  avg_width_range = np.array([avg_center[0] - avg_width / 2, avg_center[0] + avg_width / 2])

  # fit to output aspect ratio
  if fit_to == 'width':
    avg_height_range = np.array([
      avg_center[1] - avg_width * output_size[1] / output_size[0] / 2,
      avg_center[1] + avg_width * output_size[1] / output_size[0] / 2
    ]).astype(np.int).clip(0, 9999)

    avg_width_range = avg_width_range.astype(np.int).clip(0, 9999)
  elif fit_to == 'height':
    avg_height_range = avg_height_range.astype(np.int).clip(0, 9999)

    avg_width_range = np.array([
      avg_center[0] - avg_height * output_size[0] / output_size[1] / 2,
      avg_center[0] + avg_height * output_size[0] / output_size[1] / 2
    ]).astype(np.int).clip(0, 9999)

  # crop image
  result_img = img[avg_height_range[0]:avg_height_range[1], avg_width_range[0]:avg_width_range[1]].copy()

  # resize image to output size
  result_img = cv2.resize(result_img, output_size)

  # visualize
  pt1 = (int(left), int(top))
  pt2 = (int(right), int(bottom))
  cv2.rectangle(img, pt1, pt2, (255, 255, 255), 3)

  cv2.imshow('img', img)
  cv2.imshow('result', result_img)
  # write video
  out.write(result_img)
  if cv2.waitKey(1) == ord('q'):
    break

# release everything
cap.release()
out.release()
cv2.destroyAllWindows()
'''