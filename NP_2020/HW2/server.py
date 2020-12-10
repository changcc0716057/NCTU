import argparse
import socket
import string
import sys
import random
import threading
import time

Users = {} # index : username, users[username][0] = password, users[username][1] = email
board_list = {} # record the existing board and its owner
post_list = {} # index : S/N, post_list[S/N][0] = boardname, post_list[S/N][1] = title, post_list[S/N][2] = content, post_list[S/N][3] = author, post_list[S/N][4] = date
post_cnt = 0
comment_list = {} # index : S/N, comment[S/N][i][1] = ith user's comment in S/N post, comment[S/N][i][0] = username
login_user_number = [None]*10000 # initial login user list,one random number to a socket
login_user_socket = {} # a socket to a login user
function_Lock = threading.Lock()

def register(client_command, client_fd, client_addr, return_message) :
    if len(client_command) != 4 : # 格式不符
        return_message = "Usage: register <username> <email> <password>\n% "
    else :
        if client_command[1] in Users : # 代表有找到相同的 username
            return_message = "Username is already used.\n% "
        else:
            Users[client_command[1]] = [client_command[3], client_command[2]]
            return_message = "Register successfully.\n% "
    client_fd.sendto(return_message.encode(), client_addr)

def login(client_command, client_fd, return_message) :
    if len(client_command) != 3 : # 格式不符
        return_message = "Usage: login <username> <password>\n% "
    else :
        if client_fd in login_user_socket: # 代表使用這個 socket 的用戶已經 login
            return_message = "Please logout first.\n% "
        else :       
            if client_command[1] in Users and Users[client_command[1]][0] == client_command[2]: # username exists and password is right
                x = random.randint(0,9999) # 隨機產生一個 token 給新 login 的 socket
                while login_user_number[x] != None :
                    x = random.randint(0,9999)
                login_user_number[x] = client_fd # list : token 對到 socket
                login_user_socket[client_fd] = client_command[1] # dict : socket 對到 username
                return_message = str(x) + "-Welcome, " + client_command[1] + ".\n% " # 將 token 包在訊息中傳回去，使用 '-' 是為了方便處理
            else :
                return_message = "Login failed.\n% "
    client_fd.sendall(return_message.encode())  

def whoami(client_command, client_fd, client_addr, return_message):
    if len(client_command) != 1: # 格式不符
        return_message = "Usage: whoami\n% "
    else:
        if login_user_number[int(client_command[1])] == None : # 如果用戶傳來的 token number 沒有對到任何 socket，代表尚未 login
            return_message = "Please login first.\n% "
        else :
            return_message = str(login_user_socket[login_user_number[int(client_command[1])]]) + "\n% "
        client_fd.sendto(return_message.encode(), client_addr)

def list_user(client_command, client_fd, return_message):
    if len(client_command) != 1: # 格式不符
        return_message = "Usage: list-user\n% "
    else:        
        maxlen = 0 # 為了對齊 找最長的 username
        for username in Users:
            maxlen = max(maxlen, len(username))
        return_message = "Name" + " " * (maxlen - 4 + 3) + "Email\n"
        for username in Users:
            return_message += str(username) + " " * (maxlen - len(username) + 3) + str(Users[username][1]) + "\n"
        return_message += "% "
    client_fd.sendall(return_message.encode())

def logout(client_command, client_fd, return_message):
    if len(client_command) != 1: # 格式不符
        return_message = "Usage: logout\n% "
    else:
        if client_fd in login_user_socket: # 代表使用這個 socket 的用戶已經 login
            return_message = "Bye, "+ login_user_socket[client_fd] +".\n% "
            tmp = login_user_number.index(client_fd) # 找到這個 socket 代表的 number
            login_user_number[tmp] = None # 將這個 number 對到的 socket 重設為 None
            del login_user_socket[client_fd]  # 將這個 socket 從 login_user_socket 中刪除
        else:
            return_message = "Please login first.\n% "
    client_fd.sendall(return_message.encode())

def create_board(client_command, client_fd, return_message):
    if len(client_command) != 2 : # 格式不符
        return_message = "Usage: create-board <name>\n% "
    else :
        if client_fd in login_user_socket: # 代表使用這個 socket 的用戶已經 login
            if client_command[1] in board_list:
                return_message = "Board already exists.\n% "
            else:
                board_list[client_command[1]] = login_user_socket[client_fd]
                return_message = "Create board successfully.\n% "
        else :
            return_message = "Please login first.\n% "
    client_fd.sendall(return_message.encode())  

def create_post(client_message, board_name, client_fd, return_message):
    global post_cnt
    localtime = time.localtime(time.time())
    timestr = str(localtime.tm_mon) + '/' + str(localtime.tm_mday)
    client_message = client_message.split(' ')
    Syntax_Error = False
    if board_name == "--title" or board_name == "--content":
        Syntax_Error = True
    try: # 預處理字串，找出 title 及 content
        loc_title = client_message.index("--title")
        loc_content = client_message.index("--content")
        title = " ".join(client_message[loc_title+1 : loc_content]).strip()
        content = " ".join(client_message[loc_content+1 : ]).strip()
    except ValueError:
        Syntax_Error = True
    if len(title) == 0 or len(content) == 0:
        Syntax_Error = True
    if Syntax_Error: # 格式不符
        return_message = "Usage: create-post <board-name> --title <title> --content <content>\n% "
    else:
        if client_fd in login_user_socket: # 代表使用這個 socket 的用戶已經 login
            if board_name in board_list:
                post_cnt += 1
                post_list[post_cnt] = [board_name, title, content, login_user_socket[client_fd], timestr]
                comment_list[post_cnt] = []
                return_message = "Create post successfully.\n% "
            else:
                return_message = "Board does not exist.\n% "
        else :
            return_message = "Please login first.\n% "        
    client_fd.sendall(return_message.encode())

def list_board(client_command, client_fd, return_message):
    if len(client_command) != 1: # 格式不符
        return_message = "Usage: list-board\n% "
    else:        
        maxlen = 0 # 為了對齊 找最長的 boardname
        cnt = 1
        for boardname in board_list:
            maxlen = max(maxlen, len(boardname))
        return_message = "Index" + " " * (maxlen - 5 + 3) + "Name" + " " * (maxlen - 4 + 3) + "Moderator\n"
        for boardname in board_list:
            return_message += str(cnt) + " " * (maxlen - len(str(cnt)) + 3) + str(boardname) + " " * (maxlen - len(boardname) + 3) + str(board_list[boardname]) + "\n"
            cnt += 1
        return_message += "% "
    client_fd.sendall(return_message.encode())

def list_post(client_command, client_fd, return_message):
    if len(client_command) != 2: #格式不符
        return_message = "Usage: list-post <board-name>\n% "
    else:
        if client_command[1] in board_list:
            maxlen = 0
            for post in post_list: # 為了對齊 找出 title, author 中最長的字串
                if post_list[post][0] == client_command[1]:
                    maxlen = max(maxlen, post, len(post_list[post][1]), len(post_list[post][3]))
            return_message = "S/N" + " " * (maxlen - 3 + 3) + "Title" + " " * (maxlen - 5 + 3) + "Author" + " " * (maxlen - 6 + 3) + "Date"
            for post in post_list:
                if post_list[post][0] == client_command[1]:
                    return_message += "\n" + str(post) + " " * (maxlen - len(str(post)) + 3) + str(post_list[post][1]) + " " * (maxlen - len(post_list[post][1]) + 3) 
                    return_message += str(post_list[post][3]) + " " * (maxlen - len(post_list[post][3]) + 3) + str(post_list[post][4])
            return_message += "\n% " 
        else:
            return_message = "Board does not exist.\n% "
    client_fd.sendall(return_message.encode())

def read(client_command, client_fd, return_message):
    isNumber = True
    try:
        post_num = int(client_command[1])
    except ValueError:
        isNumber = False
    if len(client_command) != 2 or (not isNumber): #格式不符
        return_message = "Usage: read <post-S/N>\n% "
    else:
        if post_num in post_list:
            return_message = "Author: " + post_list[post_num][3] + '\n' + "Title: " + post_list[post_num][1] + '\n' + "Date: " + post_list[post_num][4] + '\n--\n'
            spilt_content = post_list[post_num][2].split('<br>')
            for content in spilt_content:
                return_message += content + '\n'
            return_message += "--"
            for comments in comment_list[post_num]:
                return_message += "\n" + comments[0] + ": " + comments[1]
            return_message += "\n% "
        else:
            return_message = "Post does not exist.\n% "
    client_fd.sendall(return_message.encode())

def delete_post(client_command, client_fd, return_message):
    isNumber = True
    try:
        post_num = int(client_command[1])
    except ValueError:
        isNumber = False
    if len(client_command) != 2 or (not isNumber): #格式不符
        return_message = "Usage: delete-post <post-S/N>\n% "
    else:
        if client_fd in login_user_socket: # 代表使用這個 socket 的用戶已經 login
            if post_num in post_list:
                if login_user_socket[client_fd] == post_list[post_num][3]:
                    return_message = "Delete successfully.\n% "
                    del post_list[post_num]
                else:
                    return_message = "Not the post owner.\n% "
            else:
                return_message = "Post does not exist.\n% "
        else:
            return_message = "Please login first.\n% "
    client_fd.sendall(return_message.encode())

def update_post(client_message, post_num, client_fd, return_message):
    Syntax_Error = False
    client_message = client_message.split(' ')
    flag = 0
    try:
        post_num = int(post_num)
        loc = client_message.index("--title")
        flag = 1 # 要修改 title
        new = " ".join(client_message[loc+1 : ]).strip()
    except ValueError:
        try:
            post_num = int(post_num)
            loc = client_message.index("--content")
            flag = 2 # 要修改 content
            new = " ".join(client_message[loc+1 : ]).strip()
        except ValueError:
            Syntax_Error = True
    if len(new) == 0:
        Syntax_Error = True
    if Syntax_Error: #格式不符
        return_message = "Usage: update-post <post-S/N> --title/content <new>\n% "
    else:
        if client_fd in login_user_socket: # 代表使用這個 socket 的用戶已經 login
            if post_num in post_list:
                if login_user_socket[client_fd] == post_list[post_num][3]:
                    post_list[post_num][flag] = new
                    return_message = "Update successfully.\n% "
                else:
                    return_message = "Not the post owner.\n% "
            else:
                return_message = "Post does not exist.\n% "
        else:
            return_message = "Please login first.\n% "
    client_fd.sendall(return_message.encode())

def comment(client_message, post_num, client_fd, return_message):
    isNumber = True
    client_message = client_message.split(' ')
    try:
        post_num = int(post_num)
    except ValueError:
        isNumber = False
    #預處理字串，用來找出 comment 字串
    comment = " ".join(client_message[2 : ]).strip()
    if (not isNumber) or len(comment) == 0: #格式不符
        return_message = "Usage: comment <post-S/N> <comment>\n% "
    else:
        if client_fd in login_user_socket: # 代表使用這個 socket 的用戶已經 login
            if post_num in post_list:
                comment_list[post_num].append([login_user_socket[client_fd], comment]) 
                return_message = "Comment successfully.\n% "
            else:
                return_message = "Post does not exist.\n% "
        else:
            return_message = "Please login first.\n% "
    client_fd.sendall(return_message.encode())           

def welcomemsg(conn):
    return_message = "********************************\n** Welcome to the BBS server. **\n********************************\n% "
    conn.sendall(return_message.encode())

def othercommand(client_fd, return_message):
    return_message = "Invalid Command, please type again.\n% "
    client_fd.sendall(return_message.encode())

def TCP_thread_fun(conn):
    welcomemsg(conn)
    while True:
        return_message = ""
        try:
            client_message = str(conn.recv(65000), encoding = 'utf-8')
        except BlockingIOError:
            continue
        client_command = client_message.strip().split()
        if client_command[0] == "exit":
            conn.close()
            break
        function_Lock.acquire()
        if client_command[0] == "login" :
            login(client_command, conn, return_message)
        elif client_command[0] == "list-user" :
            list_user(client_command, conn, return_message)
        elif client_command[0] == "logout" :
            logout(client_command, conn, return_message)
        elif client_command[0] == "create-board" :
            create_board(client_command, conn, return_message)
        elif client_command[0] == "create-post" :
            create_post(client_message, client_command[1], conn, return_message)
        elif client_command[0] == "list-board" :
            list_board(client_command, conn, return_message) 
        elif client_command[0] == "list-post" :
            list_post(client_command, conn, return_message) 
        elif client_command[0] == "read" :
            read(client_command, conn, return_message) 
        elif client_command[0] == "delete-post" :
            delete_post(client_command, conn, return_message) 
        elif client_command[0] == "update-post" :
            update_post(client_message, client_command[1], conn, return_message) 
        elif client_command[0] == "comment" :
            comment(client_message, client_command[1], conn, return_message)  
        else :
            othercommand(conn, return_message)
        function_Lock.release()

def UDP_thread_fun(UDP_fd):
    while True:
        return_message = ""
        try:
            client_message, client_addr = UDP_fd.recvfrom(65000)
        except BlockingIOError:
            continue
        client_message = str(client_message, encoding = 'utf-8').strip()
        client_command = client_message.split()
        function_Lock.acquire()
        if client_command[0] == "register" :
            register(client_command, UDP_fd, client_addr, return_message)
        elif client_command[0] == "whoami" :
            whoami(client_command, UDP_fd, client_addr, return_message)
        function_Lock.release()

if __name__ == "__main__":
    #get the standard input argument and set port
    parser = argparse.ArgumentParser()
    parser.add_argument('Port', type = int, help = 'Enter a connection port')
    args = parser.parse_args()
    HOST = '127.0.0.1'
    PORT = args.Port

    #socket using TCP
    server_TCP = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_TCP.bind((HOST, PORT))
    server_TCP.listen(50)
    #socket using UDP
    server_UDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_UDP.bind((HOST, PORT))

    threads = []
    UDP_thread = threading.Thread(target=UDP_thread_fun, args=(server_UDP,))
    UDP_thread.start()

    while True :
        conn, addr = server_TCP.accept()
        conn.setblocking(False)
        threads.append(threading.Thread(target=TCP_thread_fun, args=(conn,)).start())
        print("New connection")