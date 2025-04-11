import socket
import time
import pandas as pd
import threading

class Udp:
    def __init__(self):
        self.data_input = None
        self.input_list = []

    def start_udp_server(self, ip, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP 소켓 생성
        sock.bind((ip, port))  # IP 주소와 포트 바인딩

        print(f"Listening on {ip}:{port}")

        while True:

            data, addr = sock.recvfrom(1024)  # 버퍼 크기는 1024
            self.data_input = data.decode('utf-8')
            # print(f"Received message: {data.decode()} from {addr}")

            # 클라이언트에게 응답 보내기 (옵션)
            sock.sendto(b"ACK", addr)
            # time.sleep(0.1)

    def data_save(self,):
        while True:
            # with input.get_lock():
            # print(input_list)
            self.input_list.append(self.data_input)
            time.sleep(0.1)

    def keyinput(self, user_num, experiment_condition):
        while True:
            key = input()
            if key != '':
                self.input_list.append(key)
                print(key, 'put')
                print(self.input_list)
            if key == 'end':
                df = pd.DataFrame(self.input_list)
                df.to_csv('./udp_data/' + str(user_num) + '_' + experiment_condition +'.csv', index=False)
                print('dataframe has been saved')

    def run(self, user_num, experiment_condition):
        server = threading.Thread(target=self.start_udp_server, args=('192.168.0.46', 8080, ))
        save_data = threading.Thread(target=self.data_save, args=())
        key_input = threading.Thread(target=self.keyinput, args=(user_num, experiment_condition))

        server.start()
        save_data.start()
        key_input.start()

        server.join()
        save_data.join()
        key_input.join()



if __name__ == "__main__":
    user_num = input('user num: ')
    experiment_condition = input('experiment condition: ')
    udp = Udp()
    udp.run(user_num, experiment_condition)
