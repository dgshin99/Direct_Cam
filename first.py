import cv2
import numpy as np

# 얼굴 인식을 위한 Haar cascades 파일 경로
cascade_path = 'haarcascades/haarcascade_frontalface_default.xml'

# Haar cascades 파일 로드
face_cascade = cv2.CascadeClassifier(cascade_path)

# open video file
video_path = 'redvelvet_red.mp4'
cap = cv2.VideoCapture(video_path)

output_size = (187, 333)  # (width, height) 휴대폰 크기에 맞는 영상 사이즈


# initialize writing video
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# fourcc값 받아오기
out = cv2.VideoWriter('%s_output.mp4' % (video_path.split('.')[0]), fourcc, cap.get(cv2.CAP_PROP_FPS), output_size)
# 인자, 파일저장이름/ 코덱(4개의 인자를 인수로 취함, 확장자 : .mp4)/ FPS 1초당 프레임, cap.get 우리가 불러온 동영상과 똑같은 프레임으로 저장/ 저장 사이즈
# 영상의 크기는 고정이고 오브젝트 트래킹 이후 box의 크기는 유동적이어서 잘 계산해야한다.

# check if the file is opened: 동영상이 제대로 로드되면 True를 반환한다.
if not cap.isOpened():
    exit()

tracker = cv2.TrackerCSRT_create()  # OPENCV_OBJECT_TRACKERS 중 적당한 csrt사용, 정확도가 높으면 속도가 느림

ret, img = cap.read()  # 첫번째 프레임이 img에 저장

# Initialize ROI selection flag and rect
selecting_roi = True
rect = None

while True:
    if selecting_roi:
        cv2.namedWindow('Select Window')  
        # 이 프로그램이 ROI에서 window를 사용하는구나 알 수 있게 이름 지정
        cv2.imshow('Select Window', img)
        # Select Window의 첫번째 프레임을 보여줘라

        # select ROI ROI를 설정하여 rect로 반환한다.
        # 코드 실행 후 이미지에서 네모 박스로 영역지정 후 space바 누르면 ROI정보가 rect에 저장됨
        rect = cv2.selectROI('Select Window', img, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow('Select Window')

        # intialize tracker 오브젝트 트래커가 img와 rect를 따라가도록 설정(첫번째 프레임과 ROI 따라)
        tracker.init(img, rect)
        selecting_roi = False

    ret, img = cap.read() # 첫번째 프레임이 img에 저장


    #비디오 프레임을 제대로 읽었다면 ret 값이 True가 되고, 실패하면 False가 된다.
    if not ret:
        break

    success, box = tracker.update(img)  # success : 성공했냐 안했냐/ box : rect형태의 데이터

    left, top, w, h = [int(v) for v in box]  # box에 있는 데이터 가져오기

    if left < 0:
      left = 0
    if top < 0:
      top = 0
    if left + w > img.shape[1]:
      w = img.shape[1] - left
    if top + h > img.shape[0]:
      h = img.shape[0] - top
      
    # 영상의 크기는 고정이고 오브젝트 트래킹 이후 box의 크기는 유동적이어서 잘 계산해야한다.
    center_x = left + w / 2
    center_y = top + h / 2

    result_top = int(center_y - output_size[1] / 4)
    result_bottom = int(center_y + output_size[1] / 4 * 3)
    result_left = int(center_x - output_size[0] / 2)
    result_right = int(center_x + output_size[0] / 2)



    if result_left < 0:
        result_left = 0
    if result_top < 0:
        result_top = 0
    if result_right > img.shape[1]:
        result_right = img.shape[1]
    if result_bottom > img.shape[0]:
        result_bottom = img.shape[0]

    result_img = img[result_top:result_bottom, result_left:result_right].copy()  # ROI지정 박스 안보이게


# 여기부터 모자이크 코드

    # 얼굴 인식을 위해 그레이스케일로 변환
    gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 검출된 얼굴에 모자이크 처리
    for (x, y, w, h) in faces:
        face_roi = result_img[y:y+h, x:x+w]
        # 모자이크 처리를 위해 얼굴 ROI를 축소
        small_roi = cv2.resize(face_roi, (10, 10), interpolation=cv2.INTER_LINEAR)
        # 축소된 ROI를 다시 확대하여 원래 크기로 복원
        mosaic_roi = cv2.resize(small_roi, (w, h), interpolation=cv2.INTER_NEAREST)
        # 모자이크 처리된 얼굴 ROI를 원본 이미지에 적용
        result_img[y:y+h, x:x+w] = mosaic_roi

        # 얼굴 ROI를 축소하고 원래 크기로 복원하는 이유 : 모자이크 처리를 위해 얼굴 ROI를 축소하고 원래 크기로
        # 복원하게 되면 얼굴 영역이 흐릿하게 만들어 식별이 어려워진다. 그래서 축소-확대 과정을 진행 


    out.write(result_img)  # 이미지를 동영상으로 저장한다.

    # result_img가 영상 밖으로 나가게 되면 에러가 난다.
    cv2.rectangle(img, pt1=(left, top), pt2=(left + w, top + h), color=(255, 255, 255), thickness=3)
    cv2.imshow('result_img', result_img)
    cv2.imshow('img', img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):
        selecting_roi = True

print(img.shape[1])
print(img.shape[0])

cap.release()  # 오픈한 cap개체를 해제
out.release()  # 오픈한 out개체를 해제
cv2.destroyAllWindows()  # 생성한 모든 윈도우를 제거