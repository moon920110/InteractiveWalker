import socket

def start_udp_server(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP 소켓 생성
    sock.bind((ip, port))  # IP 주소와 포트 바인딩

    print(f"Listening on {ip}:{port}")

    while True:
        data, addr = sock.recvfrom(1024)  # 버퍼 크기는 1024
        print(f"Received message: {data.decode()} from {addr}")

        # 클라이언트에게 응답 보내기 (옵션)
        sock.sendto(b"ACK", addr)

if __name__ == "__main__":
    start_udp_server('192.168.0.46', 8080)