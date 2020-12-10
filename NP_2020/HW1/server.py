import argparse
import socket
import string
import sys
import select
import sqlite3
import random

client_command = []
return_message = ""
login_user_number = [None]*10000 # initial login user list,one random number to a socket
login_user_socket = {} # a socket to a login user


def register(client_command, client_socket, client_addr) :
    if len(client_command) != 4 : # 格式不符
        return_message = "Usage: register <username> <email> <password>\n% "
    else :
        cur = database_conn.cursor() 
        sql = "SELECT username from USER_INFORMATION where username = '" + client_command[1] + "' ;"
        cur.execute(sql)
        rows = cur.fetchall()
        if len(rows) != 0 : # 代表有找到相同的 username
            return_message = "Username is already used.\n% "
        else:
            database_conn.execute("INSERT INTO USER_INFORMATION(username,email,password) VALUES(?,?,?);",(client_command[1],client_command[2],client_command[3]))
            database_conn.commit()
            return_message = "Register successfully.\n% "
    client_socket.sendto(return_message.encode(), client_addr)

def login(client_command, client_socket) :
    if len(client_command) != 3 : # 格式不符
        return_message = "Usage: login <username> <password>\n% "
    else :
        if client_socket in login_user_socket: # 代表使用這個 socket 的用戶已經 login
            return_message = "Please logout first.\n% "
        else :
            cur = database_conn.cursor()
            sql = "SELECT username, password from USER_INFORMATION where username = '" + client_command[1] + "' ;"
            cur.execute(sql)
            rows = cur.fetchall()        
            if len(rows) == 0 or rows[0][1] != client_command[2]: # username is wrong or password is wrong
                return_message = "Login failed.\n% "
            else :
                x = random.randint(0,9999) # 隨機產生一個 token 給新 login 的 socket
                while login_user_number[x] != None :
                    x = random.randint(0,9999)
                login_user_number[x] = client_socket # list : token 對到 socket
                login_user_socket[client_socket] = client_command[1] # dict : socket 對到 username
                return_message = str(x) + "-Welcome, " + client_command[1] + ".\n% " # 將 token 包在訊息中傳回去，使用 '-' 是為了方便處理
    client_socket.sendall(return_message.encode())  

def whoami(client_command, client_socket, client_addr):
    if login_user_number[int(client_command[1])] == None : # 如果用戶傳來的 token number 沒有對到任何 socket，代表尚未 login
        return_message = "Please login first.\n% "
    else :
        return_message = str(login_user_socket[login_user_number[int(client_command[1])]]) + "\n% "
    client_socket.sendto(return_message.encode(), client_addr)

def list_user(client_socket):
    cur = database_conn.cursor()
    cur.execute("SELECT username, email from USER_INFORMATION")
    rows = cur.fetchall()
    maxlen = 0 # 為了對齊 找最長的 username
    for row in rows:
        maxlen = max(maxlen, len(row[0]))
    return_message = "Name" + " " * (maxlen - 4 + 2) + "Email\n"
    for row in rows:
        return_message += str(row[0]) + " " * (maxlen - len(row[0]) + 2) + str(row[1]) + "\n"
    return_message += "% "
    client_socket.sendall(return_message.encode())

def logout(client_socket):
    if client_socket in login_user_socket: # 代表使用這個 socket 的用戶已經 login
        return_message = "Bye, "+ login_user_socket[client_socket] +".\n% "
        tmp = login_user_number.index(client_socket) # 找到這個 socket 代表的 number
        login_user_number[tmp] = None # 將這個 number 對到的 socket 重設為 None
        del login_user_socket[client_socket]  # 將這個 socket 從 login_user_socket 中刪除
    else:
        return_message = "Please login first.\n% "
    client_socket.sendall(return_message.encode())

def welcomemsg(conn):
    return_message = "********************************\n** Welcome to the BBS server. **\n********************************\n% "
    conn.sendall(return_message.encode())

def othercommand(client_socket):
    return_message = "Invalid Command, please type again.\n% "
    client_socket.sendall(return_message.encode())

#get the standard input argument and set port
parser = argparse.ArgumentParser()
parser.add_argument('Port', type = int, help = 'Enter a connection port')
args = parser.parse_args()
HOST = '127.0.0.1'
PORT = args.Port

#create a SQLite database
database_conn = sqlite3.connect('user_information.db')
database_conn.execute('''CREATE TABLE IF NOT EXISTS USER_INFORMATION
                     (UID INTEGER PRIMARY KEY AUTOINCREMENT,
                     username TEXT NOT NULL UNIQUE,
                     email    TEXT NOT NULL,
                     password TEXT NOT NULL);''')

#socket using TCP
server_TCP = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_TCP.bind((HOST, PORT))
server_TCP.listen(50)
#socket using UDP
server_UDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_UDP.bind((HOST, PORT))

input_socket = []

input_socket.append(server_UDP)
input_socket.append(server_TCP)

while True:
    readable, _, outgoing_msg = select.select(input_socket, [], input_socket) # 監聽在這些 list 中的 socket，當有訊息傳入時，便回傳有反應的 socket
    for now_socket in readable : # 只考慮有訊息傳入的 list
        if now_socket is server_TCP : # 負責接收新用戶
            conn, addr = server_TCP.accept()
            conn.setblocking(False)
            input_socket.append(conn) # 將用來跟用戶接收及傳遞訊息的 socket 放進監聽的序列中 
            print("New connection")
            welcomemsg(conn)
        elif now_socket is server_UDP : # 舊用戶以 UDP 傳送訊息
            client_message, client_addr = now_socket.recvfrom(4096)
            client_message = str(client_message, encoding = 'utf-8').strip()
            client_command = client_message.split()
            if client_command[0] == "register" :
                register(client_command, now_socket, client_addr)
            elif client_command[0] == "whoami" :
                whoami(client_command, now_socket, client_addr)
        else : # 舊用戶以 TCP 傳送訊息
            client_message = str(now_socket.recv(4096), encoding = 'utf-8')
            client_command = client_message.strip().split()
            if client_command[0] == "exit":
                input_socket.remove(now_socket)
                now_socket.close()
            elif client_command[0] == "login" :
                login(client_command, now_socket)
            elif client_command[0] == "list-user" :
                list_user(now_socket)
            elif client_command[0] == "logout" :
                logout(now_socket)
            else :
                othercommand(now_socket)