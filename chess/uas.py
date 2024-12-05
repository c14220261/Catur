
import pygame
import chess
import random
import math

# Initialize the chessboard and move history
board = chess.Board()
move_history = []

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 600
SQ_SIZE = HEIGHT // 8
LABEL_SIZE = 40
INFO_PANEL_WIDTH = 200
MOVE_HISTORY_COL_WIDTH = 100

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)


# Load and resize images (replace with your own paths)
def load_and_resize_image(path):
    img = pygame.image.load(path)
    return pygame.transform.scale(img, (SQ_SIZE, SQ_SIZE))


pawn_white_img = load_and_resize_image('assets/images/white pawn.png')
rook_white_img = load_and_resize_image('assets/images/white rook.png')
knight_white_img = load_and_resize_image('assets/images/white knight.png')
bishop_white_img = load_and_resize_image('assets/images/white bishop.png')
queen_white_img = load_and_resize_image('assets/images/white queen.png')
king_white_img = load_and_resize_image('assets/images/white king.png')

pawn_black_img = load_and_resize_image('assets/images/black pawn.png')
rook_black_img = load_and_resize_image('assets/images/black rook.png')
knight_black_img = load_and_resize_image('assets/images/black knight.png')
bishop_black_img = load_and_resize_image('assets/images/black bishop.png')
queen_black_img = load_and_resize_image('assets/images/black queen.png')
king_black_img = load_and_resize_image('assets/images/black king.png')


# Function to draw the chessboard
def draw_board(screen):
    colors = [WHITE, BLACK]
    for r in range(8):
        for c in range(8):
            color = colors[(r + c) % 2]
            pygame.draw.rect(screen, color,
                             pygame.Rect(c * SQ_SIZE + LABEL_SIZE, r * SQ_SIZE + LABEL_SIZE, SQ_SIZE, SQ_SIZE))

    # Draw row and column labels
    font = pygame.font.SysFont(None, 24)
    for i in range(8):
        # Draw column labels (A-H)
        col_label = font.render(chr(ord('A') + i), True, BLACK)
        screen.blit(col_label, ((i + 0.5) * SQ_SIZE + LABEL_SIZE - col_label.get_width() // 2, 0))

        # Draw row labels (1-8)
        row_label = font.render(str(8 - i), True, BLACK)
        screen.blit(row_label, (0, (i + 0.5) * SQ_SIZE + LABEL_SIZE - row_label.get_height() // 2))


# Function to draw pieces
def draw_pieces(screen, board):
    piece_images = {
        chess.PAWN: {chess.WHITE: pawn_white_img, chess.BLACK: pawn_black_img},
        chess.ROOK: {chess.WHITE: rook_white_img, chess.BLACK: rook_black_img},
        chess.KNIGHT: {chess.WHITE: knight_white_img, chess.BLACK: knight_black_img},
        chess.BISHOP: {chess.WHITE: bishop_white_img, chess.BLACK: bishop_black_img},
        chess.QUEEN: {chess.WHITE: queen_white_img, chess.BLACK: queen_black_img},
        chess.KING: {chess.WHITE: king_white_img, chess.BLACK: king_black_img},
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            piece_img = piece_images.get(piece.piece_type).get(piece.color)
            screen.blit(piece_img, pygame.Rect(chess.square_file(square) * SQ_SIZE + LABEL_SIZE,
                                               (7 - chess.square_rank(square)) * SQ_SIZE + LABEL_SIZE, SQ_SIZE,
                                               SQ_SIZE))


# Function to get player's move input based on mouse click
def get_player_move():
    from_square = None
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                file = (x - LABEL_SIZE) // SQ_SIZE
                rank = 7 - ((y - LABEL_SIZE) // SQ_SIZE)  # Convert to chess rank (0-7 from bottom to top)
                if 0 <= file < 8 and 0 <= rank < 8:  # Ensure click is within the board
                    clicked_square = chess.square(file, rank)
                    if from_square is None:
                        # Select piece
                        if board.piece_at(clicked_square):
                            from_square = clicked_square
                    else:
                        # Make move
                        move = chess.Move(from_square, clicked_square)
                        if move in board.legal_moves:
                            board.push(move)
                            move_history.append(move)  # Add move to history
                            return move
                        else:
                            from_square = None  # Invalid move, reset


# Monte Carlo Tree Search (MCTS) Implementation
class MCTS:
    def __init__(self, board, simulations=1000):
        self.board = board
        self.simulations = simulations

    def select(self, node):
        # Selection using UCB1
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                node = node.best_child()
        return node

    def expand(self, node):
        move = node.untried_moves.pop()
        next_board = node.board.copy()
        next_board.push(move)
        child_node = MCTSNode(next_board, move, parent=node)
        node.children.append(child_node)
        return child_node

    def simulate(self, node):
        board = node.board.copy()
        while not board.is_game_over():
            legal_moves = list(board.legal_moves)
            move = random.choice(legal_moves)
            board.push(move)
        result = board.result()
        if result == "1-0":
            return 1 if self.board.turn == chess.WHITE else -1
        elif result == "0-1":
            return -1 if self.board.turn == chess.WHITE else 1
        else:
            return 0

    def backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent

    def best_move(self):
        root = MCTSNode(self.board)
        for _ in range(self.simulations):
            node = self.select(root)
            result = self.simulate(node)
            self.backpropagate(node, result)
        return root.best_child().move


class MCTSNode:
    def __init__(self, board, move=None, parent=None):
        self.board = board
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_moves = list(board.legal_moves)

    def is_terminal(self):
        return self.board.is_game_over()

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.wins / child.visits) + c_param * math.sqrt(2 * math.log(self.visits) / child.visits)
            if child.visits > 0 else float('inf')
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]


# Function for AI to make a move using MCTS
def make_ai_move(board, simulations=1000):
    mcts = MCTS(board, simulations=simulations)
    best_move = mcts.best_move()
    board.push(best_move)
    move_history.append(best_move)  # Add AI move to history


# Function to display the turn indicator
def display_turn(screen, turn):
    font = pygame.font.SysFont(None, 36)
    text_color = GREEN if turn == "Your Turn" else BLACK
    text = font.render(turn, True, text_color)
    screen.blit(text, (20, HEIGHT + LABEL_SIZE + 10))


# Function to display the move history on screen
def display_move_history(screen, move_history, start_index):
    font = pygame.font.SysFont(None, 24)
    text_color = BLACK
    max_moves_per_col = (HEIGHT - LABEL_SIZE) // 20  # Number of moves per column based on height and move text height
    col_count = 0  # Start with the first column

    for i in range(start_index, len(move_history)):
        move_text = str(move_history[i])
        text_render = font.render(move_text, True, text_color)

        x = WIDTH + LABEL_SIZE + 20 + col_count * MOVE_HISTORY_COL_WIDTH
        y = LABEL_SIZE + (i % max_moves_per_col) * 20

        screen.blit(text_render, (x, y))

        # Move to the next column if the current column is full
        if (i + 1) % max_moves_per_col == 0:
            col_count += 1


# Main game loop
def play_game(screen):
    screen.fill(WHITE)
    draw_board(screen)
    draw_pieces(screen, board)
    pygame.display.flip()

    player_turn = True
    move_history_start_index = 0

    while not board.is_game_over():
        screen.fill(WHITE)
        draw_board(screen)
        draw_pieces(screen, board)
        turn_indicator = "Your Turn" if player_turn else "AI's Turn"
        display_turn(screen, turn_indicator)
        display_move_history(screen, move_history, move_history_start_index)  # Display move history with scrolling
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    move_history_start_index = max(0, move_history_start_index - 1)  # Scroll up
                elif event.key == pygame.K_DOWN:
                    move_history_start_index = min(len(move_history) - 1, move_history_start_index + 1)  # Scroll down

        if player_turn:
            player_move = get_player_move()
            player_turn = False
        else:
            make_ai_move(board)
            player_turn = True

        # Check if the game is over
        if board.is_checkmate():
            print("Checkmate!")
            break
        elif board.is_stalemate():
            print("Stalemate!")
            break
        elif board.is_insufficient_material():
            print("Insufficient material!")
            break


# Set up the screen
screen = pygame.display.set_mode((WIDTH + INFO_PANEL_WIDTH + 2 * LABEL_SIZE, HEIGHT + 2 * LABEL_SIZE))
pygame.display.set_caption("Chess with MCTS AI")

# Run the game
play_game(screen)
