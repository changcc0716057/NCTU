#!/usr/bin/python3
import socket
import argparse
import sys
import string

received_number = 0

parser = argparse.ArgumentParser()
parser.add_argument('IP', type = str, help = 'Enter a IP address\n')
parser.add_argument('Port', type = int, help = 'Enter a connection port\n')
args = parser.parse_args()

host = args.IP
port = args.Port
addr = (host, port)

TCP_fd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
TCP_fd.connect(addr)
UDP_fd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_message = str(TCP_fd.recv(65000), encoding='utf-8')
print(server_message,end='')

while True:
    command = sys.stdin.readline().strip()
    tmp = command.split()
    if tmp[0] == "register" or tmp[0] == "whoami":
        if tmp[0] == "whoami":
            command += " " + str(received_number)
        UDP_fd.sendto(command.encode(), addr)
        server_message, tmp_addr = UDP_fd.recvfrom(65000)
        print(str(server_message, encoding = 'utf-8'), end='')
    elif tmp[0] == "exit":
        TCP_fd.sendall(command.encode())
        TCP_fd.close()
        UDP_fd.close()
        break      
    else:
        TCP_fd.sendall(command.encode())
        server_message = str(TCP_fd.recv(65000), encoding= 'utf-8')
        if tmp[0] == "login" and server_message[0] != 'U' and server_message[0] != 'P' and server_message[0] != 'L':
            tmp_msg = server_message.split('-')
            received_number = int(tmp_msg[0])
            print(tmp_msg[1], end='')
        else :
            print(server_message, end='')
