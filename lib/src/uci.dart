import 'dart:convert';
import 'dart:io';
import 'package:bishop/bishop.dart';
import 'mcts3.dart' as mc;

class UciEngine {
  late Game game;
  mc.Mcts? mcts;
  bool isSearching = false;

  UciEngine() {
    game = Game(); // Initialize a new standard chess game
  }

  void startLoop() {
    stdin
        .transform(Utf8Decoder())
        .transform(const LineSplitter())
        .listen((line) {
      handleCommand(line);
    });
  }

  void handleCommand(String command) {
    final parts = command.split(' ');
    final cmd = parts[0];

    switch (cmd) {
      case 'uci':
        stdout.writeln('id name MyDartMctsEngine');
        stdout.writeln('id author YourName');
        stdout.writeln('uciok');
        break;
      case 'isready':
        stdout.writeln('readyok');
        break;
      case 'ucinewgame':
        game = Game(); // Reset the game state
        mcts = mc.Mcts(game);
        break;
      case 'position':
        handlePosition(parts);
        break;
      case 'go':
        handleGo(parts);
        break;
      case 'stop':
        // A way to stop the search, e.g., by setting a flag
        isSearching = false;
        break;
      case 'quit':
        exit(0);
      default:
        // Ignore unknown commands
        break;
    }
  }

  void handlePosition(List<String> parts) {
    int movesIndex = parts.indexOf('moves');
    String fen = '';

    if (parts[1] == 'startpos') {
      fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
    } else if (parts[1] == 'fen') {
      fen = parts
          .sublist(2, movesIndex != -1 ? movesIndex : parts.length)
          .join(' ');
    }

    game = Game(variant: null);
    game.setup(fen: fen);
    mcts = mc.Mcts(game);

    if (movesIndex != -1) {
      List<String> moves = parts.sublist(movesIndex + 1);
      for (String moveString in moves) {
        Move? move = game.getMove(moveString);
        if (move != null) {
          game.makeMove(move);
        }
      }
    }
  }

  void handleGo(List<String> parts) async {
    // Parse time limits from the command, e.g., 'go movetime 1000'
    int moveTime = 5000; // Default to 5 seconds
    int movetimeIndex = parts.indexOf('movetime');
    if (movetimeIndex != -1 && movetimeIndex + 1 < parts.length) {
      moveTime = int.tryParse(parts[movetimeIndex + 1]) ?? 5000;
    }

    isSearching = true;

    // Run MCTS with the specified time limit. You may need to modify your Mcts.search method to accept a time limit.
    mc.EngineResult result = mcts!.monteCarlo(game);
    isSearching = false;

    if (result.hasMove) {
      stdout.writeln('bestmove ${game.toAlgebraic(result.move!)}');
    }
  }
}

void main() {
  final uciEngine = UciEngine();
  uciEngine.startLoop();
}
