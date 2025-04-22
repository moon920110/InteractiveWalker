# import cv2
# import sys
#
# # Load Web Camera
# user_num = input('user num: ')
# experiment_condition = input('experiment condition: ')
# cap = cv2.VideoCapture(0)  # load WebCamera
# if not (cap.isOpened()):
#     print("File isn't opend!!")
#
# # Set Video File Property
# videoFileName = './STS_data/'+ user_num + '_' +experiment_condition + '.avi'
# w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # width
# h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # height
# fps = cap.get(cv2.CAP_PROP_FPS)  # frame per second
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # fourcc
# delay = round(1000 / fps)  # set interval between frame
#
# # Save Video
# out = cv2.VideoWriter(videoFileName, fourcc, fps, (w, h))
# if not (out.isOpened()):
#     print("File isn't opend!!")
#     cap.release()
#     sys.exit()
#
# # Load frame and Save it
# while (True):  # Check Video is Available
#     ret, frame = cap.read()  # read by frame (ret=TRUE/FALSE)s
#
#     if ret:
#         inversed = cv2.flip(frame, 1)  # inversed frame
#
#         out.write(inversed)  # save video frame
#
#         cv2.imshow('Inversed VIDEO', inversed)
#
#
#         if cv2.waitKey(delay) == 27:  # wait 10ms until user input 'esc'
#             break
#     else:
#         print("ret is false")
#         break
#
# cap.release()  # release memory  # release memory
# cv2.destroyAllWindows()  # destroy All Window

import cv2
import sys
import time
import os

# 사용자 입력
user_num = input('user num: ')
experiment_condition = input('experiment condition: ')
video_dir = './STS_data'
os.makedirs(video_dir, exist_ok=True)

# 웹캠 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera is not opened!")
    sys.exit()

# 기본 설정
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
save_fps = 30  # 저장용 fps 고정
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 코덱
videoFileName = os.path.join(video_dir, f"{user_num}_{experiment_condition}.avi")
out = cv2.VideoWriter(videoFileName, fourcc, save_fps, (w, h))

# 녹화 시작
print("Recording started. Press ESC to stop.")
start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    frame_flipped = cv2.flip(frame, 1)
    out.write(frame_flipped)  # 파일로 저장
    cv2.imshow('Camera View', frame_flipped)

    # ESC 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_count += 1

# 종료 후 정보 출력
duration = time.time() - start_time
print(f"Recording finished. Duration: {duration:.2f}s, Frames: {frame_count}, Approx. FPS: {frame_count / duration:.2f}")

cap.release()
out.release()
cv2.destroyAllWindows()