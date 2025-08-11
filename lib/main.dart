import 'bishop.dart';

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
