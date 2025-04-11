import cv2
import numpy as np

def track_marker(video_path):
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)

    # 색상 범위 정의 (예: 빨간색)
    lower_green = np.array([50, 100, 50])
    upper_green = np.array([70, 255, 255])

    while True:
        ret, frame = cap.read()  # 프레임 읽기
        if not ret:
            break

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
                # 중심에 작은 원 그리기
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

        # 결과 이미지 표시
        cv2.imshow('Frame', frame)

        # 'q'를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    track_marker('test.MOV')