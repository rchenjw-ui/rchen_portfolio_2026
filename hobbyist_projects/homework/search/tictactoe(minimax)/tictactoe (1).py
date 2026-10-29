"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state(): #DONE
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board): #DONE
    x_count = 0
    o_count = 0
    for row in board:
        for space in row:
            if space == "X":
                x_count += 1
            if space == "O":
                o_count += 1
    assert x_count == o_count or x_count == o_count +1
    if x_count > o_count:
        return O
    if x_count == o_count:
        return X
    #note: if O odd, return O. If x odd, return X
    #issue: even even. Will have to handle all 4 parity cases
    #switch from parity to less or greater
    #if x > o, o. X cannot be less than o
    #DONE!
    """
    Returns player who has the next turn on a board.
    """



def actions(board): #DONE given board, return actions, find all empty spaces
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    #needs to be a set? Need to learn about sets and implementation with sets
    out = []
    row_index = 0
    for row in board:
        space_index = 0
        for space in row:
            if space == EMPTY:
                out.append((row_index,space_index))
            space_index += 1
        row_index += 1
    return(set(out))

def result(board, action): # done
    #needs exceptions, for one, needs to say if it's out of bounds
    board_copy = copy.deepcopy(board)
    x = action[0]
    y = action[1]
    if board_copy[x][y] != EMPTY:
        raise ValueError
    if x not in range(0,3) or y not in range(0,3):
        raise ValueError
    board_copy[x][y] = player(board_copy)
    #issue: apparently, the board is modified?
    return(board_copy)
    """
    Returns the board that results from making move (i, j) on the board.
    """



def winner(board):
    #check for full row or column
    #
    """
    Returns the winner of the game, if there is one.
    """
    #row detector
    for c in board:
        if c == [O,O,O]:
            return O
        elif c == [X, X, X]:
            return X
    #col detector:
    #see board[0]
    for c in range(0,3):
        if board[0][c] == board[1][c] and board[1][c] == board[2][c]:
            if board[0][c] == X:
                return X
            if board[0][c] == O:
                return O
    #detect diagonal:
    if board[0][0] == board[1][1] == board[2][2]:
        if board[0][0] == X:
            return X
        if board[0][0] == O:
            return O
    if board[0][2] == board[1][1] == board[2][0]:
        if board[0][2] == X:
            return X
        if board[0][2] == O:
            return O
    return None


def terminal(board):
    #need to update check
    if utility(board) != 0:
        return True
    if len(actions(board)) == 0:
        return True
    else:
        return False

    #done


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == None:
        return 0
    if winner(board) == X:
        return 1
    if winner(board) == O:
        return -1

#for minimax, may need to implement a node data structure

class Node:
    def __init__(self, parent, action):
        try:
            self.board = result(parent.board, action)
        except:
            if action == None:
                self.board = parent #this is to handle the action == None, it's for the root node
            else:
                self.board = result(parent, action)
            #this is the killer line: unless a node before it is used, it will never update
        self.parent = parent
        self.action = action
        self.player = player(self.board)
        self.value = None
        self.children = []

def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    This is the one to be working on as of 3/4/2026
    """

    """
    FOR 3/7/2026:
    - need to resolve the root node bug, because apparently, the root node inevitably freezes it. And, also the pop condition for non_terminal
    - Pop condition: May need to default pop; if it's not terminal it gets caught in a loop
    """
    #anomalous error: seems like the minmax algorithm didn't select a minimum value child
    if terminal(board) == True:
        return None
    frontier = generate_tree(board)
    print(frontier)
    #print("INIT")
    explored = []
    cont = True
    count = 0
    while cont == True:
        #note: some values are "none" for certain children
        print(len(frontier))
        print("one round")
        try:
            chosen = frontier[0]
            print(chosen)
            frontier = frontier[1:]
            parent = chosen.parent
        except:
            if len(frontier) == 0:
                return None
            raise ValueError
        else:
            print(chosen.board == board)
            #control flow issue: If parents aren't a node, do not add parent to node
             #this is to see what happens when all 5 are exhausted and it is sent back up

            #note: needs a type check, only add if not a list
            #problem: parent isn't a node, how to handle? This is the issue
            #also, root node: it needs a root tag so the algorithm doesn't just follow it into oblivion
            #note: if it's the root node, and parent == None, do not add
            if parent not in frontier and parent not in explored:
                if type(parent) != type([]):
                    frontier.append(parent)
            #value eval
            if terminal(chosen.board) == True:
                if type(parent) != type([]):
                    if chosen not in parent.children:
                        parent.children.append(chosen)
                chosen.value = utility(chosen.board)
                #print(chosen.value)
                explored.append(chosen)
                #print("explored")
            else:
                #print(f"{len(chosen.children)} children")
                #consistently no children, need to add children
                #minmax logic
                #print(len(actions(chosen.board)))
                if len(chosen.children) == len(actions(chosen.board)): #completeness check
                    #print("complete")
                    #print(chosen.children)
                    values = [c.value for c in chosen.children] #values
                    print(values)
                    #issue: for some, the value of children may just be none
                    #how to solve? if children are none
                    #remove chosen if complete and value clean
                    explored.append(chosen)
                    if None not in values:
                        #this is a brute force solve to avoid assigning a value before all are cleaned up.
                        if chosen.player == X:
                            chosen.value = max(values)
                        else:
                            chosen.value = min(values)
                        if type(parent) != type([]):
                            if chosen not in parent.children:
                                parent.children.append(chosen)
            if chosen.board == board and len(chosen.children) == len(actions(chosen.board)):
                #need to change terminate conditions and see if it's the root node
                #issue: sometimes it terminates at the original, other times it terminates at the optimal board
                #terminate condition
                #seems not to deploy at len == 1
                cont = False
                print("last one")
                #maybe use chosen?
                #now, you have children of optimal, pick optimal move
                for i in chosen.board:
                    print(i)
                print(chosen.player)
                print("children")
                #issue: we have the optimal board already? What?
                children = chosen.children
                values_dict = {}
                for c in children:
                    values_dict[c.value] = c
                    for i in c.board:
                        print(i)
                    print(c.value)
                    print(c.player)
                #final minmax
                if player(chosen.board) == X:
                    optimal_value = max(values_dict.keys())
                if player(chosen.board) == O:
                    optimal_value = min(values_dict.keys())
                #print(children)
                print(values_dict)
                return(values_dict[optimal_value].action)





def generate_tree(board):
    #note: change to while loop, check if frontier empty
    #create parent node
    parent_board = Node(parent = board, action = None)
    #print(parent_board)
    #issue: needs action to exist, but on the other hand, it's just a container
    initial = actions(board)
    frontier = []
    terminal_nodes = []
    for c in initial:
        node = Node(parent = parent_board, action = c)
        board_check = node.board
        if terminal(board_check) == False:
            frontier.append(node)
        if terminal(board_check) == True:
            terminal_nodes.append(node)
    #issue: need a way to iterate through a list, without having issues
    #solution: queue-like frontier, while loop
    count = 0
    while len(frontier) != 0:
        item = frontier[0]
        candidates = actions(item.board)
        for c in candidates:
            node = Node(parent = item, action = c)
            board_check = node.board
            #issue: when board is created, it's not automatically updated by action
            if terminal(board_check) == False:
                frontier.append(node)
            else:
                terminal_nodes.append(node)
        try:
            frontier = frontier[1:]
        except:
            if len(frontier) == 0:
                return terminal_nodes
    return terminal_nodes







def all_terminal(boards):
    for c in boards:
        if terminal(board) == False:
            return False
    return True




if __name__ == "__main__":
    board = [
        [X, O, O],
        [X, X, EMPTY],
        [EMPTY, EMPTY, EMPTY]
    ]
    print(player(board))

    ##confirmed normal board handling BUT WHAT? I cannot find any "list calling"
    final = minimax(board)
    print("DONE")
    print(final)
    #error: seems minmax doesn't spot the optimal move, it seems like it let two moves play out for some reason
