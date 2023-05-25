import cv2

# 얼굴 인식을 위한 Haar cascades 파일 경로
cascade_path = 'haarcascades/haarcascade_frontalface_default.xml'

# Haar cascades 파일 로드
face_cascade = cv2.CascadeClassifier(cascade_path)

# 스티커 이미지 로드
sticker_path = 'sticker.jpg'
sticker_img = cv2.imread(sticker_path, cv2.IMREAD_UNCHANGED)

# 동영상 파일 불러오기
video_path = 'redvelvet_red.mp4'
cap = cv2.VideoCapture(video_path)

output_size = (187, 333)  # 출력 영상 크기

# 동영상 저장을 위한 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), output_size)

# 첫번째 프레임을 가져옴
ret, img = cap.read()

# ROI 영역 지정
cv2.namedWindow('Select Window')
cv2.imshow('Select Window', img)
rect = cv2.selectROI('Select Window', img, fromCenter=False, showCrosshair=True)
cv2.destroyWindow('Select Window')

while True:
    ret, img = cap.read()
    if not ret:
        break

    # 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 얼굴 인식
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # 얼굴 위치에 스티커 추가
        sticker_resized = cv2.resize(sticker_img, (w, h))
        x_offset = x
        y_offset = y
        if sticker_resized.shape[2] == 4:  # 스티커 이미지에 알파 채널이 있는 경우
            alpha_s = sticker_resized[:, :, 3] / 255.0  # 알파 채널 가져오기
        else:
            alpha_s = None
        if alpha_s is not None:  # 알파 채널이 있는 경우
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                img[y_offset:y_offset + h, x_offset:x_offset + w, c] = (alpha_s * sticker_resized[:, :, c] +
                                                                        alpha_l * img[y_offset:y_offset + h,
                                                                                      x_offset:x_offset + w, c])
        else:  # 알파 채널이 없는 경우
            for c in range(0, 3):
                img[y_offset:y_offset + h, x_offset:x_offset + w, c] = sticker_resized[:, :, c]

    # 초점 영상 재생
    result_img = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]].copy()
    result_img = cv2.resize(result_img, output_size)
    cv2.imshow('Video', result_img)

    # 동영상 저장
    out.write(result_img)

    if cv2.waitKey(1) == ord('q'):
        break

# 종료
cap.release()
out.release()
cv2.destroyAllWindows()
