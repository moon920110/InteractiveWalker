import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

def track_marker(participant_num):
    # 비디오 캡처 객체 생성
    trajectory_list = []
    participant_data_list = [participant_num+'_human.avi', participant_num+'_N.avi', participant_num+'_P.avi', participant_num+'_F.avi']

    for data in participant_data_list:
        trajectory = []
        video_path = 'STS_data/' + data
        cap = cv2.VideoCapture(video_path)

        # 색상 범위 정의 (예: 빨간색)
        lower_green = np.array([50, 100, 40])
        upper_green = np.array([90, 255, 255])

        while True:
            ret, frame = cap.read()  # 프레임 읽기
            if not ret:
                break
            frame = frame[:, 500:1400]

            # BGR에서 HSV 색공간으로 변환
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 색상 범위에 따라 마스크 생성
            mask = cv2.inRange(hsv, lower_green, upper_green)

            # 마스크에서 윤곽선 찾기
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # 최소 크기 설정
                if cv2.contourArea(contour) > 50:
                    # 윤곽선의 경계 사각형 구하기
                    x, y, w, h = cv2.boundingRect(contour)
                    # 사각형 그리기
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # 중심 좌표 구하기
                    center_x = x + w // 2
                    center_y = y + h // 2
                    trajectory.append([x, 1080-y])
                    # 중심에 작은 원 그리기
                    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

            # 결과 이미지 표시
            cv2.imshow('Frame', frame)

            # 'q'를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        trajectory = np.array(trajectory)
        mean = np.mean(trajectory, axis=0)
        distances = np.linalg.norm(trajectory - mean, axis=1)

        # 2. 이상치 기준 설정 (예: 평균 + 2.0 * 표준편차 이상은 이상치로 간주)
        threshold = np.mean(distances) + 1.0 * np.std(distances)
        mask = distances < threshold

        # 3. 이상치 제거
        trajectory = trajectory[mask]
        cap.release()
        cv2.destroyAllWindows()
        trajectory_list.append(trajectory)



    plt.figure(figsize=(8, 6))
    coeffs = np.polyfit(np.array(trajectory_list[0])[:, 0], np.array(trajectory_list[0])[:, 1], deg=3)
    poly = np.poly1d(coeffs)
    x_fit = np.linspace(min(np.array(trajectory_list[0])[:, 0]), max(np.array(trajectory_list[0])[:, 0]), 300)
    y_fit = poly(x_fit)
    plt.plot(x_fit, y_fit, 'o--', label=participant_data_list[0], color='black')
    base_x_fit = x_fit[0]
    base_y_fit = y_fit[0]

    coeffs = np.polyfit(np.array(trajectory_list[1])[:, 0], np.array(trajectory_list[1])[:, 1], deg=3)
    poly = np.poly1d(coeffs)
    x_fit = np.linspace(min(np.array(trajectory_list[1])[:, 0]), max(np.array(trajectory_list[1])[:, 0]), 300)
    y_fit = poly(x_fit)
    x_fit = x_fit + (base_x_fit - x_fit[0])
    y_fit = y_fit + (base_y_fit - y_fit[0])
    plt.plot(x_fit, y_fit, 'o--', label=participant_data_list[1], color='blue')

    coeffs = np.polyfit(np.array(trajectory_list[2])[:, 0], np.array(trajectory_list[2])[:, 1], deg=3)
    poly = np.poly1d(coeffs)
    x_fit = np.linspace(min(np.array(trajectory_list[2])[:, 0]), max(np.array(trajectory_list[2])[:, 0]), 300)
    y_fit = poly(x_fit)
    x_fit = x_fit + (base_x_fit - x_fit[0])
    y_fit = y_fit + (base_y_fit - y_fit[0])
    plt.plot(x_fit, y_fit, 'o--', label=participant_data_list[2], color='red')

    coeffs = np.polyfit(np.array(trajectory_list[3])[:, 0], np.array(trajectory_list[3])[:, 1], deg=3)
    poly = np.poly1d(coeffs)
    x_fit = np.linspace(min(np.array(trajectory_list[3])[:, 0]), max(np.array(trajectory_list[3])[:, 0]), 300)
    y_fit = poly(x_fit)
    x_fit = x_fit + (base_x_fit - x_fit[0])
    y_fit = y_fit + (base_y_fit - y_fit[0])
    plt.plot(x_fit, y_fit, 'o--', label=participant_data_list[3], color='green')
    # plt.plot(np.array(trajectory_list[0])[:, 0], np.array(trajectory_list[0])[:, 1], 'o--',
    #          label=participant_data_list[1], color='black')  # 원래 점
    # plt.plot(np.array(trajectory_list[1])[:, 0], np.array(trajectory_list[1])[:, 1], 'o--', label=participant_data_list[1], color='blue')# 원래 점
    # plt.plot(np.array(trajectory_list[2])[:, 0], np.array(trajectory_list[2])[:, 1], 'o--', label=participant_data_list[2], color='red')
    # plt.plot(np.array(trajectory_list[3])[:, 0], np.array(trajectory_list[3])[:, 1], 'o--', label=participant_data_list[3], color='green')  # 원래 점

    # plt.plot(x_smooth, y_smooth, '-', label=participant_data_list[i], color='blue')  # 스무딩 곡선
    plt.title("Smoothed 2D Curve")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    print('hi')

if __name__ == "__main__":
    track_marker(str(2))