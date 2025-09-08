import serial
import time

# Arduino가 연결된 COM 포트. 리눅스나 MacOS 사용자는 '/dev/ttyUSB0' 등으로 변경
arduino = serial.Serial(port='/dev/ttyACM0', baudrate=115200, timeout=1)

def send_data(data):
    arduino.write(data.encode())  # 데이터를 바이트로 인코딩하여 보냄

def read_data():
    while True:
        if arduino.inWaiting() > 0:
            line = arduino.readline().decode('utf-8').strip()  # 데이터 읽기
            print("Received:", line)
            return line

# 지속적인 통신을 위한 메인 루프
while True:
    cmd = input("Enter 'ping' or other command: ")
    send_data(cmd)  # 입력 받은 명령어를 Arduino로 보냄
    # read_data()  # Arduino의 응답을 기다림
    time.sleep(1)  # 다음 명령어 보내기 전에 간단한 지연