#!/usr/bin/python3
import argparse
import socket
import string
import sys

received_number = 0

parser = argparse.ArgumentParser()
parser.add_argument('IP', type = str, help = 'Enter a IP address\n')
parser.add_argument('Port', type = int, help = 'Enter a connection port\n')
args = parser.parse_args()

HOST = args.IP
PORT = args.Port
#socket using TCP
client_TCP = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_TCP.connect((HOST, PORT))
server_message = str(client_TCP.recv(4096), encoding='utf-8')
print(server_message,end='')
#socket using UDP 
client_UDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
addr = (HOST, PORT)

while(True):
    command = sys.stdin.readline().strip()
    tmp = command.split()
    if tmp[0] == "register" or tmp[0] == "whoami":
        if tmp[0] == "whoami":
            command += " " + str(received_number)
        client_UDP.sendto(command.encode(), addr)
        server_message, tmp_addr = client_UDP.recvfrom(4096)
        print(str(server_message, encoding = 'utf-8'), end='')
    elif tmp[0] == "exit":
        client_TCP.sendall(command.encode())
        client_TCP.close()
        client_UDP.close()
        break
    else:
        client_TCP.sendall(command.encode())
        server_message = str(client_TCP.recv(4096), encoding= 'utf-8')
        if tmp[0] == "login" and server_message[0] != 'U' and server_message[0] != 'P' and server_message[0] != 'L':
            tmp_msg = server_message.split('-')
            received_number = int(tmp_msg[0])
            print(tmp_msg[1], end='')
        elif tmp[0] == "list-user":
            print(server_message, end='')
        else :
            print(server_message, end='')
