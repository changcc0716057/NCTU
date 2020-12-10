#!/usr/bin/python3
import socket
import argparse
import sys
import string
import select
import time
import threading
received_number = 0
login_user_name = ""
chatroom_close = True
chatroom_port = {} # index : name, chatroom_port[index] = port
chatroom_records = {} # index : chatroom name, chatroom_records[index][i] = msg
chatroom_semaphore = threading.Semaphore(0)
TCP_fd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
UDP_fd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def welcomemsg(client_fd):
    return_message = "*****************************\n** Welcome to the chatroom **\n*****************************\n"
    msg_num = len(chatroom_records[login_user_name])
    if msg_num < 3 and msg_num > 0:
        for i in range(0, msg_num - 1):
            return_message += chatroom_records[login_user_name][i] + "\n"
        return_message += chatroom_records[login_user_name][msg_num-1]
    elif msg_num >= 3:
        for i in range(msg_num - 3, msg_num - 1):
            return_message += chatroom_records[login_user_name][i] + "\n"
        return_message += chatroom_records[login_user_name][msg_num-1]
    client_fd.sendall(return_message.encode())

def Recv(sckfd):
    while True:
        try:
            return str(sckfd.recv(65000), encoding = 'utf-8')
        except BlockingIOError:
            continue

def check(status, sckfd):
    Recv(sckfd)
    sckfd.sendall(status.encode())

def create_chatroom(port):
    global chatroom_close
    ADDR = ('127.0.0.1', port)

    #socket using TCP
    chatroom_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    chatroom_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # 1 代表將 SO_REUSEADDR 標記為 true，OS 會在 server socket 被關閉後馬上釋放 port
    chatroom_server.bind(ADDR)
    chatroom_server.listen(10)
    chatroom_semaphore.release()

    input_sck = [chatroom_server]
    close = False

    while not close:
        readable, _, _ = select.select(input_sck, [], [])

        for sckfd in readable : #考慮有訊息傳入的 list

            if sckfd is chatroom_server : # 負責接收新用戶
                conn, _ = chatroom_server.accept()
                conn.setblocking(False)
                input_sck.append(conn) 
                username = Recv(conn) # client_msg 為新用戶的名字
                welcomemsg(conn)

                if username != login_user_name : # boardcast
                    localtime = time.localtime(time.time()) # 讀取 sys msg 發送的時間
                    sys_msg = "sys [" + str(localtime.tm_hour) + ":" + str(localtime.tm_min) + "] : " + username + " join us."
                    for sck in input_sck :
                        if sck != conn and sck != chatroom_server : 
                            sck.sendall(sys_msg.encode())
                            check("open", sck)

            else : #舊用戶傳送訊息
                client_msg = str(sckfd.recv(65000), encoding='utf-8').strip()
                msg = client_msg.split()

                if msg[0] == login_user_name and msg[1] == "leave-chatroom" :
                    close = True
                    for sck in input_sck : # boardcast
                        if sck != sckfd and sck != chatroom_server :
                            localtime = time.localtime(time.time()) # 讀取 sys msg 發送的時間
                            sys_msg = "sys [" + str(localtime.tm_hour) + ":" + str(localtime.tm_min) + "] : the chatroom is close."
                            sck.sendall(sys_msg.encode())
                            check("close", sck)
                            client_msg = Recv(sck)
                            if client_msg == "Bye":
                                sck.close()
                                input_sck.remove(sck)     
                    sckfd.close()
                    input_sck.remove(sckfd)
                    break

                elif msg[0] == login_user_name and msg[1] == "detach" :
                    sckfd.close()
                    input_sck.remove(sckfd)

                elif msg[0] != login_user_name and msg[1] == "leave-chatroom" :
                    localtime = time.localtime(time.time()) # 讀取 sys msg 發送的時間
                    sys_msg = "sys [" + str(localtime.tm_hour) + ":" + str(localtime.tm_min) + "] : " + msg[0] + " leave us."
                    for sck in input_sck :
                        if sck != sckfd and sck != chatroom_server : 
                            sck.sendall(sys_msg.encode())
                            check("open", sck)
                    sckfd.close()
                    input_sck.remove(sckfd)

                else :
                    for sck in input_sck : # boardcast
                        if sck != sckfd and sck != chatroom_server :
                            sck.sendall(client_msg.encode())
                            check("open", sck)
                    chatroom_records[login_user_name].append(client_msg)

    chatroom_server.close()
    input_sck.remove(chatroom_server)
    TCP_fd.sendall(("close " + login_user_name).encode())
    chatroom_close = True

def join_chatroom_or_attach(port, command_type):
    addr = ('127.0.0.1', port)

    fd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    fd.connect(addr)

    myname = str(login_user_name)
    fd.sendall(myname.encode())
    server_message = str(fd.recv(65000), encoding='utf-8').strip()
    print(server_message)

    input_sck = [fd, sys.stdin]
    close = False

    while not close:
        readable, _, _ = select.select(input_sck, [], [])

        for sckfd in readable:
            if sckfd == fd:
                server_message = str(fd.recv(65000), encoding='utf-8')
                print(server_message)
                fd.sendall("Received".encode())
                server_message = str(fd.recv(65000), encoding='utf-8')
                if server_message == "close":
                    fd.sendall("Bye".encode())
                    fd.close()
                    close = True
                    break
            
            else:
                user_msg = sys.stdin.readline().strip() 
                localtime = time.localtime(time.time()) # 讀取 msg 發送的時間
                if user_msg == "leave-chatroom":
                    msg = login_user_name + " " + user_msg
                    fd.sendall(msg.encode())
                    fd.close()
                    close = True
                    break

                elif command_type == "attach" and user_msg == "detach":
                    msg = login_user_name + " " + user_msg
                    fd.sendall(msg.encode())
                    fd.close()
                    close = True
                    break

                else:
                    msg = login_user_name
                    msg += " [" + str(localtime.tm_hour) + ":" + str(localtime.tm_min) + "] : "
                    msg += user_msg
                    fd.sendall(msg.encode())
    
    print("Welcome back to BBS.\n% ", end='')


if __name__ == "__main__":
    # Get Address
    parser = argparse.ArgumentParser()
    parser.add_argument('IP', type = str, help = 'Enter a IP address\n')
    parser.add_argument('Port', type = int, help = 'Enter a connection port\n')
    args = parser.parse_args()

    addr = (args.IP, args.Port)
    TCP_fd.connect(addr)

    server_message = str(TCP_fd.recv(65000), encoding='utf-8') # welcome msg
    print(server_message,end='')

    while True:
        command = sys.stdin.readline().strip()
        tmp = command.split()

        if tmp[0] == "register" or tmp[0] == "whoami" or tmp[0] == "list-chatroom":
            if tmp[0] != "register" :
                command += " " + str(received_number)
            UDP_fd.sendto(command.encode(), addr)
            server_message, tmp_addr = UDP_fd.recvfrom(65000)
            print(str(server_message, encoding = 'utf-8'), end='')
        
        elif tmp[0] == "exit":
            TCP_fd.sendall(command.encode())
            TCP_fd.close()
            UDP_fd.close()
            break     

        elif tmp[0] == "create-chatroom" :
            TCP_fd.sendall(command.encode())
            server_message = str(TCP_fd.recv(65000), encoding= 'utf-8')
            print(server_message, end='')
            if server_message[0] == 's':
                # initialize some data
                port = int(tmp[1])
                chatroom_records[login_user_name] = []
                chatroom_port[login_user_name] = port
                chatroom_close = False

                # create chatroom server thread
                chatroom_server_thread = threading.Thread(target=create_chatroom, args=(chatroom_port[login_user_name],))
                chatroom_server_thread.start()

                # owner connect to chatroom server
                chatroom_semaphore.acquire()
                join_chatroom_or_attach(port, "attach")

        elif tmp[0] == "join-chatroom" : 
            TCP_fd.sendall(command.encode())
            server_message = str(TCP_fd.recv(65000), encoding= 'utf-8')
            if server_message[0] != 'P' and server_message[0] != 'T' and server_message[0] != 'U':
                join_chatroom_or_attach(int(server_message), "join-chatroom")
            else:
                print(server_message, end='')
            
        elif tmp[0] == "restart-chatroom" :
            TCP_fd.sendall(command.encode())
            server_message = str(TCP_fd.recv(65000), encoding= 'utf-8')
            if server_message[0] == 's':
                server_message = server_message.split('\n')
                print(server_message[0])

                # initialize some data
                chatroom_close = False

                # create chatroom server thread
                chatroom_server_thread = threading.Thread(target=create_chatroom, args=(int(server_message[1]),))
                chatroom_server_thread.start()

                # owner connect to chatroom server
                chatroom_semaphore.acquire()
                join_chatroom_or_attach(chatroom_port[login_user_name], "attach")
            else:
                print(server_message, end='')           

        elif tmp[0] == "attach" :
            if login_user_name == "":
                print("Please login first.\n% ", end='')
            else:
                if chatroom_port[login_user_name] == 0:
                    print("Please create-chatroom first.\n% ", end='')
                else:
                    if chatroom_close :
                        print("Please restart-chatroom first.\n% ", end='')
                    else:
                        join_chatroom_or_attach(chatroom_port[login_user_name], "attach")
            
        else:
            TCP_fd.sendall(command.encode())
            server_message = str(TCP_fd.recv(65000), encoding= 'utf-8')
            if tmp[0] == "login" and server_message[0] != 'U' and server_message[0] != 'P' and server_message[0] != 'L':
                tmp_msg = server_message.split('-')
                print(tmp_msg[1], end='')

                #initialize some datas
                received_number = int(tmp_msg[0])
                login_user_name = tmp[1]
                chatroom_port[login_user_name] = 0

            elif tmp[0] == "logout":
                print(server_message, end='')
                if server_message[0] == 'B':
                    login_user_name = ""
                    received_number = 0

            else :
                print(server_message, end='')
