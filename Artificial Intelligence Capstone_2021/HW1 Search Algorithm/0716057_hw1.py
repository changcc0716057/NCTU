import argparse
import queue
import sys
import time
import copy
import heapq
import itertools

Search_Type = ["BFS", "DFS", "IDS", "A*", "IDA*"]
Exist_State = {} # index = Board(int), value = [predecessor, (idx, x_loc, y_loc), (depth/cost)]
counter = itertools.count() # use to make sure every entry in the priority queue is unique
maxsize = 1 # use to record the maximum number of nodes kept in memory
push = 1    # use to record the number of pushing nodes into queue/stack/priority queue
pop = 0     # use to record the number of popping nodes into queue/stack/priority queue

class Car:
    def __init__(self, idx, x_loc, y_loc, length, direction):
        self._idx = idx              # index of car
        self._x_loc = x_loc          # location of x
        self._y_loc = y_loc          # location of y
        self._length = length        # length of car
        self._direction = direction  # direction of car: 1 is horizontal; 2 is vertical.
    
    @property
    def Index(self):
        return self._idx

    @property
    def X_loc(self):
        return self._x_loc

    @property
    def Y_loc(self):
        return self._y_loc

    @property
    def Length(self):
        return self._length
    @property
    def Direction(self):
        return self._direction   

    @X_loc.setter
    def X_loc(self, value:int):
        self._x_loc = value

    @Y_loc.setter
    def Y_loc(self, value:int):
        self._y_loc = value

class Board_State:
    def __init__(self, cars, board):
        self._cars = cars    # list of Car
        self._board = board  # 6x6 board
        self._key = 0
    
    @property
    def Cars(self)->list:
        return self._cars
    
    @property
    def Board(self)->list:
        return self._board
    
    @property
    def Key(self)->int:
        return self._key

    def State_to_Key(self)->int:
        ret = 0
        for car in self._cars:
            ret = ret * 100 + car.X_loc * 10 + car.Y_loc
        return ret

    @Key.setter
    def Key(self, value:int):
        self._key = value

    def MoveLeft(self, idx, x_loc, y_loc, length):
        new_state = copy.deepcopy(self)
        new_state.Board[x_loc][y_loc-1] = True
        new_state.Board[x_loc][y_loc+length-1] = False
        new_state.Cars[idx].Y_loc = new_state.Cars[idx].Y_loc - 1
        new_state.Key = new_state.State_to_Key()
        return new_state

    def MoveRight(self, idx, x_loc, y_loc, length):
        new_state = copy.deepcopy(self)
        new_state.Board[x_loc][y_loc+length] = True
        new_state.Board[x_loc][y_loc] = False
        new_state.Cars[idx].Y_loc = new_state.Cars[idx].Y_loc + 1
        new_state.Key = new_state.State_to_Key()
        return new_state
    
    def MoveUp(self, idx, x_loc, y_loc, length):
        new_state = copy.deepcopy(self)
        new_state.Board[x_loc-1][y_loc] = True
        new_state.Board[x_loc+length-1][y_loc] = False
        new_state.Cars[idx].X_loc = new_state.Cars[idx].X_loc - 1
        new_state.Key = new_state.State_to_Key()
        return new_state

    def MoveDown(self, idx, x_loc, y_loc, length):
        new_state = copy.deepcopy(self)
        new_state.Board[x_loc+length][y_loc] = True
        new_state.Board[x_loc][y_loc] = False
        new_state.Cars[idx].X_loc = new_state.Cars[idx].X_loc + 1
        new_state.Key = new_state.State_to_Key()
        return new_state

class PriorityQueue:

    def __init__(self):
        self._queue = []
        self._size = 0
    
    def push(self, entry):
        heapq.heappush(self._queue, entry)
        self._size += 1

    def pop(self):
        try:
            ret = heapq.heappop(self._queue)
        except IndexError:
            print("Index Error happens because of trying popping an item \
            from an empty heap.")
        else:
            self._size -= 1
            return ret

    def top(self):
        try:
            return self._queue[0]
        except IndexError:
            print("Index Error happens because of trying accessing the top item \
            from an empty heap.")
    
    def empty(self)->bool:
        return (self._size == 0)

    def size(self)->int:
        return self._size

def ReadFile(FileName: str)->Board_State:
    cars = []
    board = [[False for i in range(6)] for j in range(6)]

    ### Store the information of the Beginning Board ###
    f = open(FileName)
    line = f.readline()
    while line:
        information = line.strip().split()
        new_car = Car(int(information[0]), int(information[1]), int(information[2]), 
                    int(information[3]), int(information[4]))
        cars.append(new_car)
        if new_car.Direction == 1:
            for i in range(new_car.Y_loc, new_car.Y_loc+new_car.Length):
                board[new_car.X_loc][i] = True
        else:
            for i in range(new_car.X_loc, new_car.X_loc+new_car.Length):
                board[i][new_car.Y_loc] = True
        line = f.readline()
    f.close()

    return Board_State(cars, board)

def BFS(initial_state: Board_State)->int:

    # Initialize Data
    global maxsize, pop, push
    q = queue.Queue(maxsize=0) # maxsize <= 0: The size of the queue is not limited
    q.put(initial_state)
    termianl_code = -1
    Exist_State[initial_state.Key] = [-1, (-1, -1, -1)]

    # Traversal
    while(not q.empty()):

        maxsize = max(maxsize, q.qsize())
        current_state = q.get()
        current_cars, current_board, current_key = current_state.Cars, current_state.Board, current_state.Key
        pop += 1

        
        for i in range(len(current_cars)):
            idx, x_loc, y_loc, length, direction  = current_cars[i].Index, current_cars[i].X_loc, current_cars[i].Y_loc, current_cars[i].Length, current_cars[i].Direction

            # Current car is Horizontal
            if direction == 1:

                # Check whether we can move current car to left
                if y_loc > 0 and not current_board[x_loc][y_loc-1]:
                    new_state = current_state.MoveLeft(idx, x_loc, y_loc, length)

                    # Check if the state can be put into Queue or not
                    if new_state.Key not in Exist_State:
                        q.put(new_state)
                        Exist_State[new_state.Key] = [current_key, (idx, new_state.Cars[idx].X_loc, new_state.Cars[idx].Y_loc)]
                        push += 1

                # Check whether we can move current car to right
                if y_loc+length < 6 and not current_board[x_loc][y_loc+length]:
                    new_state = current_state.MoveRight(idx, x_loc, y_loc, length)

                    # Check if the state can be put into Queue or not
                    if new_state.Key not in Exist_State:
                        q.put(new_state)
                        Exist_State[new_state.Key] = [current_key, (idx, new_state.Cars[idx].X_loc, new_state.Cars[idx].Y_loc)]
                        push += 1

                    maxsize = max(maxsize, q.qsize())

                    # Check whether we find the final state
                    if new_state.Cars[0].X_loc == 2 and new_state.Cars[0].Y_loc == 4:
                        termianl_code = new_state.Key
                        break

            # Current car is Vertical
            else:

                # Check whether we can move current car to Up
                if x_loc > 0 and not current_board[x_loc-1][y_loc]:
                    new_state = current_state.MoveUp(idx, x_loc, y_loc, length)

                    # Check if the state can be put into Queue or not
                    if new_state.Key not in Exist_State:
                        q.put(new_state)
                        Exist_State[new_state.Key] = [current_key, (idx, new_state.Cars[idx].X_loc, new_state.Cars[idx].Y_loc)]
                        push += 1

                # Check whether we can move current car to Down #
                if x_loc+length < 6 and not current_board[x_loc+length][y_loc]:
                    new_state = current_state.MoveDown(idx, x_loc, y_loc, length)

                    # Check if the state can be put into Queue or not
                    if new_state.Key not in Exist_State:
                        q.put(new_state)
                        Exist_State[new_state.Key] = [current_key, (idx, new_state.Cars[idx].X_loc, new_state.Cars[idx].Y_loc)]
                        push += 1

        # Find the final state
        if termianl_code != -1:
            break

    return termianl_code

def DFS(initial_state: Board_State, depth: int)->int:

    # Initialize Data
    global maxsize, push, pop
    stack = []
    stack.append([initial_state, 0])
    termianl_code = -1
    Exist_State[initial_state.Key] = [-1, (-1, -1, -1), 0]
    
    # Traversal
    while(len(stack) != 0):

        maxsize = max(maxsize, len(stack))
        top = stack.pop()
        current_state, current_depth = top[0], top[1]
        current_cars, current_board, current_key = current_state.Cars, current_state.Board, current_state.Key
        pop += 1
        
        for i in range(len(current_cars)-1, -1, -1):

            idx, x_loc, y_loc, length, direction  = current_cars[i].Index, current_cars[i].X_loc, current_cars[i].Y_loc, current_cars[i].Length, current_cars[i].Direction
            
            # Current car is Horizontal
            if direction == 1:

                # Check whether we can move current car to left
                if y_loc > 0 and not current_board[x_loc][y_loc-1]:
                    new_state = current_state.MoveLeft(idx, x_loc, y_loc, length)

                    # Check if the state can be put into Stack or not
                    if  ((new_state.Key not in Exist_State) and (current_depth < depth or depth == -1)) or \
                        ((new_state.Key in Exist_State) and (current_depth+1 < Exist_State[new_state.Key][2]) and (depth != -1)):
                        stack.append([new_state, current_depth+1])
                        Exist_State[new_state.Key] = [current_key, (idx, new_state.Cars[idx].X_loc, new_state.Cars[idx].Y_loc), current_depth+1]
                        push += 1
                    
                # Check whether we can move current car to Right  
                if y_loc+length < 6 and not current_board[x_loc][y_loc+length]:
                    new_state = current_state.MoveRight(idx, x_loc, y_loc, length)

                    # Check if the state can be put into Stack or not
                    if  ((new_state.Key not in Exist_State) and (current_depth < depth or depth == -1)) or \
                        ((new_state.Key in Exist_State) and (current_depth+1 < Exist_State[new_state.Key][2]) and (depth != -1)):
                        stack.append([new_state, current_depth+1])
                        Exist_State[new_state.Key] = [current_key, (idx, new_state.Cars[idx].X_loc, new_state.Cars[idx].Y_loc), current_depth+1]
                        push += 1

                    maxsize = max(maxsize, len(stack))

                    # Find the terminal state
                    if new_state.Cars[0].X_loc == 2 and new_state.Cars[0].Y_loc == 4:
                        Exist_State[new_state.Key] = [current_key, (idx, new_state.Cars[idx].X_loc, new_state.Cars[idx].Y_loc), current_depth+1]
                        termianl_code = new_state.Key
                        break

            # Current car is Vertical
            else:

                # Check whether we can move current car to Up
                if x_loc > 0 and not current_board[x_loc-1][y_loc]:
                    new_state = current_state.MoveUp(idx, x_loc, y_loc, length)

                    # Check if the state can be put into Stack or not
                    if  ((new_state.Key not in Exist_State) and (current_depth < depth or depth == -1)) or \
                        ((new_state.Key in Exist_State) and (current_depth+1 < Exist_State[new_state.Key][2]) and (depth != -1)):
                        stack.append([new_state, current_depth+1])
                        Exist_State[new_state.Key] = [current_key, (idx, new_state.Cars[idx].X_loc, new_state.Cars[idx].Y_loc), current_depth+1]
                        push += 1

                # Check whether we can move current car to Down
                if x_loc+length < 6 and not current_board[x_loc+length][y_loc]:
                    new_state = current_state.MoveDown(idx, x_loc, y_loc, length)
                    
                    # Check if the state can be put into Stack or not
                    if  ((new_state.Key not in Exist_State) and (current_depth < depth or depth == -1)) or \
                        ((new_state.Key in Exist_State) and (current_depth+1 < Exist_State[new_state.Key][2]) and (depth != -1)):
                        stack.append([new_state, current_depth+1])
                        Exist_State[new_state.Key] = [current_key, (idx, new_state.Cars[idx].X_loc, new_state.Cars[idx].Y_loc), current_depth+1]
                        push += 1

        # Find the final state
        if termianl_code != -1:
            break

    return termianl_code

def IDS(initial_state: Board_State)->int:

    global maxsize, push, pop
    depth = 0
    # try to find answer with depth_limit = 0
    terminal_state = DFS(initial_state, depth)

    # terminal state == -1 implies that we still haven't found the goal
    while(terminal_state == -1):
        maxsize = 1
        push = 1
        pop = 0
        # increase depth_limit by 1
        depth += 1
        Exist_State.clear()
        # try to find answer with new depth_limit
        terminal_state = DFS(initial_state, depth)

    return terminal_state

def A_star(initial_state: Board_State)->int:

    # blocking heuristic 
    def Heuristic(third_row: list, start: int)->int:
        cnt = 0
        for i in range(start, 6):
            if third_row[i] == 0:
                cnt += 1
        return cnt

    # check if we can push the new state to priority queue
    def checker(current_state: Board_State, new_state: Board_State, current_depth: int, idx: int):
        global push
        new_cost = current_depth + 1 + Heuristic(new_state.Board[2], new_state.Cars[0].Y_loc + new_state.Cars[0].Length)

        # 若尚未被發現或已經被發現但是 cost 更低的話，可以放入 Priority Queue
        if (new_state.Key not in Exist_State) or \
            (new_state.Key in Exist_State and new_cost < Exist_State[new_state.Key][2]):

            Entry = [new_cost, next(counter), new_state, current_depth+1]
            OpenList.push(Entry)
            Exist_State[new_state.Key] = [current_state.Key, (idx, new_state.Cars[idx].X_loc, new_state.Cars[idx].Y_loc), new_cost]  
            push += 1

        return                      

    global maxsize, pop
    OpenList = PriorityQueue()
    Entry = [Heuristic(initial_state.Board[2], initial_state.Cars[0].Y_loc + initial_state.Cars[0].Length), next(counter), initial_state, 0]
    OpenList.push(Entry)
    terminal_code = -1
    Exist_State[initial_state.Key] = [-1, (-1, -1, -1), Heuristic(initial_state.Board[2], initial_state.Cars[0].Y_loc + initial_state.Cars[0].Length)]

    # Traversal
    while(not OpenList.empty()):

        maxsize = max(maxsize, OpenList.size())
        item = OpenList.pop()
        priorty, current_state, current_depth = item[0], item[2], item[3]
        current_cars, current_board = current_state.Cars, current_state.Board

        # check if the node is in closed list
        if priorty > Exist_State[current_state.Key][2]:
            continue
        pop += 1
        
        for i in range(len(current_cars)):
            idx, x_loc, y_loc, length, direction  = current_cars[i].Index, current_cars[i].X_loc, current_cars[i].Y_loc, current_cars[i].Length, current_cars[i].Direction

            # Current car is Horizontal
            if direction == 1:

                # Check whether we can move current car to left
                if y_loc > 0 and not current_board[x_loc][y_loc-1]: 
                    new_state = current_state.MoveLeft(idx, x_loc, y_loc, length)
                    checker(current_state, new_state, current_depth, idx)

                # Check whether we can move current car to Right 
                if y_loc+length < 6 and not current_board[x_loc][y_loc+length]:
                    new_state = current_state.MoveRight(idx, x_loc, y_loc, length)
                    checker(current_state, new_state, current_depth, idx)

                    maxsize = max(maxsize, OpenList.size())

                    # Find the terminal state
                    if new_state.Cars[0].X_loc == 2 and new_state.Cars[0].Y_loc == 4: 
                        terminal_code = new_state.Key
                        break

            # Current car is Vertical
            else:

                # Check whether we can move current car to Up
                if x_loc > 0 and not current_board[x_loc-1][y_loc]: 
                    new_state = current_state.MoveUp(idx, x_loc, y_loc, length)
                    checker(current_state, new_state, current_depth, idx)

                # Check whether we can move current car to Down
                if x_loc+length < 6 and not current_board[x_loc+length][y_loc]:
                    new_state = current_state.MoveDown(idx, x_loc, y_loc, length)
                    checker(current_state, new_state, current_depth, idx)

        # Find the final state
        if terminal_code != -1:
            break

    return terminal_code

def DFS_with_heuristic(initial_state: Board_State, limit_cost: int)->int:

    # blocking heuristic     
    def Heuristic(third_row: list, start: int)->int:
        cnt = 0
        for i in range(start, 6):
            if third_row[i] == 0:
                cnt += 1
        return cnt

    # check if we can push the new state to stack   
    def checker(current_state: Board_State, new_state: Board_State, current_depth: int, idx: int):
        nonlocal next_limit_cost
        global push
        new_cost = current_depth + 1 + Heuristic(new_state.Board[2], new_state.Cars[0].Y_loc + new_state.Cars[0].Length)
        
        # 若 new state 尚未被發現過且 cost 小於 limit cost，可以放入 stack
        # 若 new state 已經被發現過且新的 cost 小於 Exist state 所記錄的 cost 的話，可以放入 stack
        if ((new_state.Key not in Exist_State) and new_cost <= limit_cost) or \
            ((new_state.Key in Exist_State) and new_cost < Exist_State[new_state.Key][2]):
            stack.append([new_state, current_depth+1])
            Exist_State[new_state.Key] = [current_key, (idx, new_state.Cars[idx].X_loc, new_state.Cars[idx].Y_loc), new_cost]
            push += 1

        # update the next cost limit
        if new_cost > limit_cost:
            next_limit_cost = min(new_cost, next_limit_cost) 

        return 

    global maxsize, pop
    next_limit_cost = 2147483647 # use to find the next cost limit
    stack = []
    stack.append([initial_state, 0])
    terminal_code = -1
    Exist_State[initial_state.Key] = [-1, (-1, -1, -1), 
    Heuristic(initial_state.Board[2], 
    initial_state.Cars[0].Y_loc + initial_state.Cars[0].Length)]

    # Traversal
    while(len(stack) != 0):

        maxsize = max(maxsize, len(stack))
        top = stack.pop()
        current_state, current_depth = top[0], top[1]
        current_cars, current_board, current_key = current_state.Cars, \
                                    current_state.Board, current_state.Key
        pop += 1

        
        for i in range(len(current_cars)-1, -1, -1):

            idx, x_loc, y_loc, length, direction  = current_cars[i].Index, \
                current_cars[i].X_loc, current_cars[i].Y_loc, current_cars[i].Length, \
                current_cars[i].Direction

            # Current car is Horizontal
            if direction == 1:

                # Check whether we can move current car to left
                if y_loc > 0 and not current_board[x_loc][y_loc-1]:
                    new_state = current_state.MoveLeft(idx, x_loc, y_loc, length)
                    checker(current_state, new_state, current_depth, idx)

                # Check whether we can move current car to Righ
                if y_loc+length < 6 and not current_board[x_loc][y_loc+length]:
                    new_state = current_state.MoveRight(idx, x_loc, y_loc, length)
                    checker(current_state, new_state, current_depth, idx)

                    maxsize = max(maxsize, len(stack))

                    # Find the terminal state
                    if new_state.Cars[0].X_loc == 2 and new_state.Cars[0].Y_loc == 4:
                        terminal_code = new_state.Key
                        break

            # Current car is Vertical
            else:

                # Check whether we can move current car to Up
                if x_loc > 0 and not current_board[x_loc-1][y_loc]:
                    new_state = current_state.MoveUp(idx, x_loc, y_loc, length)
                    checker(current_state, new_state, current_depth, idx)

                # Check whether we can move current car to Down
                if x_loc+length < 6 and not current_board[x_loc+length][y_loc]:
                    new_state = current_state.MoveDown(idx, x_loc, y_loc, length)
                    checker(current_state, new_state, current_depth, idx)

        # Find the final state
        if terminal_code != -1:
            break
    
    if terminal_code == -1:
        terminal_code *= next_limit_cost

    return terminal_code

def IDA_star(initial_state: Board_State)->int:
    global maxsize, push, pop
    limit_cost = 1
    # try to find answer with cost_limit = 1
    terminal_state = DFS_with_heuristic(initial_state, limit_cost)

    # terminal state == -1 implies that we still haven't found the goal
    while(terminal_state <= 0):
        maxsize = 1
        push = 1
        pop = 0
        # update the new cost_limit
        limit_cost = -1 * terminal_state
        Exist_State.clear()
        # try to find answer with new cost_limit
        terminal_state = DFS_with_heuristic(initial_state, limit_cost)

    return terminal_state   

def main(args):
    global counter, maxsize, push, pop
    print("Searching Algorithm:", Search_Type[args.Type-1], "\nLevel:", args.FileName[:len(args.FileName)-4])
    initial_state = ReadFile(args.FileName)
    initial_state.Key = initial_state.State_to_Key()
    index_num = 0 # 用來儲存 final state
    Exist_State.clear()
    counter = itertools.count()
    maxsize = 1
    push = 1
    pop = 0

    begin = time.time()
    if args.Type == 1:
        index_num = BFS(initial_state)
    elif args.Type == 2:
        index_num = DFS(initial_state, -1)
    elif args.Type == 3:
        index_num = IDS(initial_state)
    elif args.Type == 4:
        index_num = A_star(initial_state)
    elif args.Type == 5:
        index_num = IDA_star(initial_state)
    finish = time.time()
    print("Spending Time:", round(finish - begin, 3), "s")
    print("Number of Expanding Nodes:", len(Exist_State))
    print("Maximum number of nodes kept in memory:", maxsize)
    print("# of pop / # of push:", pop/push, "(", pop, "/", push, ")")

    # save solution by visiting predecessor
    Solution = []
    while index_num != -1:
        Solution.append(Exist_State[index_num][1])
        index_num = Exist_State[index_num][0]
    Solution.reverse()
    Solution.pop(0)

    # print solution
    print("Solution Step:", len(Solution), "steps")
    for i in range(0, len(Solution)):
        print("step", i+1, ": [",  Solution[i][0], ",", Solution[i][1], ",", Solution[i][2], "]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('Type', type = int, 
    help = 'Enter type of searching\n1.BFS2.DFS 3.IDS 4.A* 5.IDA*)\n')
    parser.add_argument('FileName', type = str, help = 'Enter a file name\n')
    args = parser.parse_args()
    main(args)