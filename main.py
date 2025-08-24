import chess.pgn
import chess.engine
import pandas as pd

# Extracts features from a single game.
def extract_features_from_game(game, engine):
    print("Extracting features from a new game.")
    features, labels = [], []
    board = game.board()
    move_number = 0
    prev_eval = 0
    move_history = []

    # Loop through each move.
    for move in game.mainline_moves():
        move_number += 1
        print(f"Processing move: {move_number}")
        piece = board.piece_at(move.from_square)

        # Get move features.
        feat = {
            "move_number": move_number,
            "piece": piece.piece_type if piece else 0,
            "from_square": move.from_square,
            "to_square": move.to_square,
            "is_capture": board.is_capture(move),
            "is_check": board.is_check(),
            "is_castle": board.is_castling(move),
            "fen": board.fen(),
        }

        # Get positional features.
        print("Calculating positional features.")
        feat["material_balance"] = material_balance(board)
        feat["developed_pieces"] = developed_pieces(board)
        feat["pst_score"] = piece_square_table(board)
        feat["opponent_threats"] = opponent_threat_count(board)

        # Get move history.
        for i, past in enumerate(move_history[-2:], 1):
            feat[f"prev_{i}_piece"] = past["piece"]
            feat[f"prev_{i}_is_capture"] = past["is_capture"]

        # Get engine evaluation.
        print("Getting engine evaluation.")
        board.push(move)
        info = engine.analyse(board, chess.engine.Limit(depth=12))
        eval_score = info["score"].white().score(mate_score=10000)

        if eval_score is None:
            eval_score = prev_eval

        # Label the move.
        diff = prev_eval - eval_score
        if diff < 20:
            label = "good"
        elif diff > 100:
            label = "blunder"
        else:
            label = "inaccuracy"

        prev_eval = eval_score
        features.append(feat)
        labels.append(label)
        move_history.append(feat)

    print(f"Finished extracting features for the game. Total moves: {len(features)}")
    print(features[-1], "->", labels[-1])
    return features, labels

# Builds the dataset from a PGN file.
def build_dataset(pgn_file, engine_path="stockfish", output_csv="chess_dataset.csv"):
    print(f"Building dataset from {pgn_file}.")
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    print("Chess engine started.")
    all_rows = []

    with open(pgn_file) as f:
        game_count = 0
        while game_count < 500:
            game = chess.pgn.read_game(f)
            if game is None:
                print("End of PGN file.")
                break
            game_count += 1
            print(f"Processing game: {game_count}")
            feats, labels = extract_features_from_game(game, engine)
            for feat, label in zip(feats, labels):
                feat["label"] = label
                all_rows.append(feat)

    print("Closing chess engine.")
    engine.quit()

    # Save the dataset to a CSV file.
    print(f"Saving dataset to {output_csv}.")
    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)
    print(f"✅ Dataset saved to {output_csv} with {len(df)} rows")

    return df

# Calculates material balance.
def material_balance(board):
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    score = 0
    for piece_type, val in values.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * val
        score -= len(board.pieces(piece_type, chess.BLACK)) * val
    return score

# Calculates developed pieces.
def developed_pieces(board):
    minors = [chess.BISHOP, chess.KNIGHT]
    dev = 0
    for piece_type in minors:
        for sq in board.pieces(piece_type, chess.WHITE):
            if sq not in [chess.B1, chess.G1, chess.C1, chess.F1]:
                dev += 1
        for sq in board.pieces(piece_type, chess.BLACK):
            if sq not in [chess.B8, chess.G8, chess.C8, chess.F8]:
                dev += 1
    return dev

# Calculates piece-square table score.
def piece_square_table(board):
    pst = {
        chess.PAWN: [0,5,5,0,5,10,50,0] * 8,
        chess.KNIGHT: [-50,-40,-30,-30,-30,-30,-40,-50] * 8
    }
    score = 0
    for piece in chess.PIECE_TYPES:
        for sq in board.pieces(piece, chess.WHITE):
            score += pst.get(piece, [0]*64)[sq]
        for sq in board.pieces(piece, chess.BLACK):
            score -= pst.get(piece, [0]*64)[chess.square_mirror(sq)]
    return score

# Calculates opponent threat count.
def opponent_threat_count(board):
    threats = 0
    for sq in chess.SQUARES:
        if board.is_attacked_by(not board.turn, sq):
            threats += 1
    return threats

# Start the script.
print("Script started.")
build_dataset("lichess_db.pgn", engine_path="/opt/homebrew/bin/stockfish")
print("Script finished.")