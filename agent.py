"""
An AI player for Othello. 
"""

import random
import sys
import time

# You can use the functions in othello_shared to write your AI
from othello_shared import find_lines, get_possible_moves, get_score, play_move

cache = {}

def eprint(*args, **kwargs): #you can use this for debugging, as it will print to sterr and not stdout
    print(*args, file=sys.stderr, **kwargs)
    
# Method to compute utility value of terminal state
def compute_utility(board, color):
    #IMPLEMENT
    curr_score = get_score(board)
    white_score = curr_score[0]
    black_score = curr_score[1]
    if color == 1:
        return white_score - black_score
    elif color == 2:
        return black_score - white_score

# Better heuristic value of board
def compute_heuristic(board, color): #not implemented, optional
    curr_score = get_score(board)
    white_score = curr_score[0]
    black_score = curr_score[1]
    if color == 1:
        return white_score - black_score
    elif color == 2:
        return black_score - white_score
    return 0

############ MINIMAX ###############################
def minimax_min_node(board, color, limit, caching = 0):

    # Check cache
    if caching == 1:
        if board in cache:
            return cache[board]

    if color == 1:
        opponent = 2
    else:
        opponent = 1 

    possible_moves = get_possible_moves(board, opponent)

    # Initializing moves and utility
    minUtility = float('inf')
    best_move = None

    if len(possible_moves) == 0 or limit == 0:
        # fixed utility in the end
        return best_move, compute_utility(board, color)

    for new_move in possible_moves:
        # Get the new board
        new_board = play_move(board, opponent, new_move[0], new_move[1])

        # Compute utility
        max_node = minimax_max_node(new_board, color, limit - 1, caching)
        # Cache then new board
        if caching == 1:
            cache[new_board] = (new_move, max_node[1])

        if max_node[1] < minUtility:
            best_move = new_move
            minUtility = max_node[1]

    return best_move, minUtility

def minimax_max_node(board, color, limit, caching = 0): #returns highest possible utility

    #verifies cache initially, if not in cache then run the rest
    if caching == 1:
        if board in cache:
            return cache[board]

    possible_moves = get_possible_moves(board, color)

    # Initializing moves and utility
    maxUtility = -float('inf')
    best_move = None

    if len(possible_moves) == 0 or limit == 0:
        # fixed utility in the end
        return best_move, compute_utility(board, color) 

    for new_move in possible_moves:
        # Get the new board
        new_board = play_move(board, color, new_move[0], new_move[1])

        # Compute utility
        #_, utility = minimax_min_node(new_board, color, limit - 1, caching)
        min_node = minimax_min_node(new_board, color, limit - 1, caching)
        # Cache then new board
        if caching == 1:
            cache[new_board] = (new_move, min_node[1])

        if min_node[1] > maxUtility:
            best_move = new_move
            maxUtility = min_node[1]
    
    return best_move, maxUtility

def select_move_minimax(board, color, limit, caching = 0):
    """
    Given a board and a player color, decide on a move. 
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.  

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic 
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.    
    """
    cache.clear()
    minimax_res = minimax_max_node(board, color, limit, caching)
    return minimax_res[0]

############ ALPHA-BETA PRUNING #####################
def alphabeta_min_node(board, color, alpha, beta, limit, caching = 0, ordering = 0):
    # Get the opponent
    if color == 1:
        opponent = 2
    else:
        opponent = 1
        
    # Check cache
    if caching == 1:
        if board in cache:
            return cache[board]

    possible_moves = get_possible_moves(board, opponent)

    # Initializing moves and utility
    minUtility = float('inf')
    best_move = None

    if len(possible_moves) == 0 or limit == 0:
        # fixed utility in the end
        return best_move, compute_utility(board, color)

    for new_move in possible_moves:
        # Get the new board
        new_board = play_move(board, opponent, new_move[0], new_move[1])

        # Compute utility
        max_node = alphabeta_max_node(new_board, color, alpha, beta, limit - 1, caching, ordering)
        # Cache then new board
        if caching == 1:
            cache[new_board] = (new_move, max_node[1])

        if max_node[1] < minUtility:
            best_move = new_move
            minUtility = max_node[1]

        # pruning
        beta = min(beta, max_node[1])
        if beta <= alpha:
            break
        
    return best_move, minUtility

def alphabeta_max_node(board, color, alpha, beta, limit, caching = 0, ordering = 0):
    if caching == 1:
        if board in cache:
            return cache[board]
    
    possible_moves = get_possible_moves(board, color)

    # Initializing moves and utility
    maxUtility = -float('inf')
    best_move = None

    if len(possible_moves) == 0 or limit == 0:
        # fixed utility in the end
        return best_move, compute_utility(board, color)

    # order the states
    if ordering == 1:
        utilities = {}
        possible_moves = []
        for move in possible_moves:
            new_board = play_move(board, color, move[0], move[1])
            successor_utility = compute_utility(new_board, color)
            if successor_utility in utilities:
                if utilities[successor_utility] != [move]:
                    utilities[successor_utility].append(move)
            else:
                utilities[successor_utility] = [move]
            ordered_utility = utilities.keys().sort()
            for utility in ordered_utility:
                possible_moves += utilities[utility]
    
    for new_move in possible_moves:
        # Get the new board
        new_board = play_move(board, color, new_move[0], new_move[1])

        # Compute utility
        min_node = alphabeta_min_node(new_board, color, alpha, beta, limit - 1, caching, ordering)
        # Cache then new board
        if caching == 1:
            cache[new_board] = (new_move, min_node[1])

        if min_node[1] > maxUtility:
            best_move = new_move
            maxUtility = min_node[1]

        # pruning
        alpha = max(alpha, min_node[1])
        if beta <= alpha:
            break

    return best_move, maxUtility

def select_move_alphabeta(board, color, limit, caching = 0, ordering = 0):
    """
    Given a board and a player color, decide on a move. 
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.  

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic 
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.    
    If ordering is ON (i.e. 1), use node ordering to expedite pruning and reduce the number of state evaluations. 
    If ordering is OFF (i.e. 0), do NOT use node ordering to expedite pruning and reduce the number of state evaluations. 
    """
    cache.clear()
    alphabeta_res  = alphabeta_max_node(board, color, -float('inf'), float('inf'), limit, caching, ordering)
    return alphabeta_res[0]

####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("Othello AI") # First line is the name of this AI
    arguments = input().split(",")
    
    color = int(arguments[0]) #Player color: 1 for dark (goes first), 2 for light. 
    limit = int(arguments[1]) #Depth limit
    minimax = int(arguments[2]) #Minimax or alpha beta
    caching = int(arguments[3]) #Caching 
    ordering = int(arguments[4]) #Node-ordering (for alpha-beta only)

    if (minimax == 1): eprint("Running MINIMAX")
    else: eprint("Running ALPHA-BETA")

    if (caching == 1): eprint("State Caching is ON")
    else: eprint("State Caching is OFF")

    if (ordering == 1): eprint("Node Ordering is ON")
    else: eprint("Node Ordering is OFF")

    if (limit == -1): eprint("Depth Limit is OFF")
    else: eprint("Depth Limit is ", limit)

    if (minimax == 1 and ordering == 1): eprint("Node Ordering should have no impact on Minimax")

    while True: # This is the main loop
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        # The first number is the score for player 1 (dark), the second for player 2 (light)
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()
        dark_score = int(dark_score_s)
        light_score = int(light_score_s)

        if status == "FINAL": # Game is over.
            print
        else:
            board = eval(input()) # Read in the input and turn it into a Python
                                  # object. The format is a list of rows. The
                                  # squares in each row are represented by
                                  # 0 : empty square
                                  # 1 : dark disk (player 1)
                                  # 2 : light disk (player 2)

            # Select the move and send it to the manager
            if (minimax == 1): #run this if the minimax flag is given
                movei, movej = select_move_minimax(board, color, limit, caching)
            else: #else run alphabeta
                movei, movej = select_move_alphabeta(board, color, limit, caching, ordering)
            
            print("{} {}".format(movei, movej))

if __name__ == "__main__":
    run_ai()
