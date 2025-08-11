import 'package:bishop/bishop.dart';

final Map pieceValues = {
  PieceType.pawn(): 1,
  PieceType.knight(): 3,
  PieceType.bishop(): 3.5,
  PieceType.rook(): 5,
  PieceType.queen(): 9,
  PieceType.king(): 10
};

// simple material based evaluation
double evaluatePosition(Game c, int player) {
  if (c.result != null) {
    if (c.drawn) {
      // draw is a neutral outcome
      return 0.0;
    } else {
      // otherwise must be a mate
      if (c.turn == player) {
        // avoid mates
        return -9999.99;
      } else {
        // go for mating
        return 9999.99;
      }
    }
  } else {
    // otherwise do a simple material evaluation
    var evaluation = 0.0;
    var sq_color = 0;

    for (var i = 0; i <= c.board.length; i++) {
      sq_color = (sq_color + 1) % 2;
      if ((i & 0x88) != 0) {
        i += 7;
        continue;
      }

      final piece = c.board[i];
      if (piece != null) {
        evaluation += (piece == player)
            ? pieceValues[piece.type]
            : -pieceValues[piece.type];
      }
    }

    return evaluation;
  }
}

void main() {
  final game = Game();
  List<Move> moves = game.generateLegalMoves();
  // print(moves.map((e) => game.toSan(e)).toList());
  print(moves.map((e) => game.toAlgebraic(e)).toList());

  // while (!game.gameOver) {
  game.makeRandomMove();
  // }
  print(game.ascii());
  print(game.state.hash);
  print(game.pgn());
}
