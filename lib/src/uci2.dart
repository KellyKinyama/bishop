// uci.dart

import 'dart:convert';
import 'dart:io';
import 'package:bishop/bishop.dart';
import 'package:dart_torch/transformer/transformer_decoder.dart';
import 'mcts4.dart' as mc;
import 'gpt_model.dart'; // Import your GPT model file

class UciEngine {
  late Game game;
  mc.Mcts? mcts;
  bool isSearching = false;
  final TransformerDecoder policyNetwork;
  final Map<String, int> stoi;
  final Map<int, String> itos;

  UciEngine(this.policyNetwork, this.stoi, this.itos) {
    game = Game();
    mcts = mc.Mcts(game, policyNetwork, stoi, itos);
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
        game = Game();
        mcts = mc.Mcts(game, policyNetwork, stoi, itos);
        break;
      case 'position':
        handlePosition(parts);
        break;
      case 'go':
        handleGo(parts);
        break;
      case 'stop':
        isSearching = false;
        break;
      case 'quit':
        exit(0);
      default:
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
    mcts = mc.Mcts(game, policyNetwork, stoi, itos);

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
    int moveTime = 5000;
    int movetimeIndex = parts.indexOf('movetime');
    if (movetimeIndex != -1 && movetimeIndex + 1 < parts.length) {
      moveTime = int.tryParse(parts[movetimeIndex + 1]) ?? 5000;
    }

    isSearching = true;

    if (mcts != null) {
      mc.EngineResult result = await mcts!.monteCarlo(game);
      isSearching = false;

      if (result.hasMove) {
        stdout.writeln('bestmove ${game.toAlgebraic(result.move!)}');
      }
    }
  }
}

void main() {
  // Load your GPT model and vocabulary here
  final result = createGptModel();
  final policyNetwork = result.$1;
  // final stoi = {}; // Your vocabulary map
  // final itos = {}; // Your reverse vocabulary map
  final stoi = result.$2 as Map<String, int>; // Explicit cast to correct type
  final itos = result.$3 as Map<int, String>; // Explicit cast to correct type

  final uciEngine = UciEngine(policyNetwork, stoi, itos);
  uciEngine.startLoop();
}
