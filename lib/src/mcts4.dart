// mcts3.dart

import 'dart:math';
import 'package:bishop/bishop.dart';
import 'package:dart_torch/nn/value.dart';
import 'package:dart_torch/nn/value_vector.dart';
import 'package:dart_torch/transformer/transformer_decoder.dart';
import 'gpt_model.dart'; // Assume this file exists and contains your GPT implementation

// Re-using the classes from your previous context
class EngineResult {
  final Move? move;
  final double? eval;
  final int? depth;
  bool get hasMove => move != null;
  const EngineResult({this.move, this.eval, this.depth});
}

class Edge {
  double reward = 0;
  int visits = 0;
  Move move;
  double actionValue = 0;
  double meanActionValue = 0;
  double priorProbability; // New field for the policy network's output
  Edge(this.move, this.priorProbability);
}

class Node {
  int key;
  int visitCount = 0;
  double totalValue = 0.0;
  List<Edge> children = [];
  Node(this.key);
  double get qValue => visitCount == 0 ? 0.0 : totalValue / visitCount;
}

class Mcts {
  Node rootNode;
  Game game;
  final TransformerDecoder policyNetwork;
  final Map<String, int> stoi;
  final Map<int, String> itos;

  Mcts(this.game, this.policyNetwork, this.stoi, this.itos)
      : rootNode = Node(game.state.hash);

  Future<EngineResult> monteCarlo(Game pos) async {
    rootNode = Node(pos.state.hash);

    for (int i = 0; i < 200; i++) {
      // 1. Selection
      List<Edge> path = [];
      Node currentNode = rootNode;
      Game searchGame = Game.fromPgn(pos.pgn());

      while (!searchGame.gameOver && currentNode.children.isNotEmpty) {
        Edge bestEdge = _selectBestChild(currentNode);
        path.add(bestEdge);
        searchGame.makeMove(bestEdge.move, false);
        currentNode = getNode(searchGame);
      }

      // 2. Expansion
      if (!searchGame.gameOver) {
        await _expandNode(currentNode, searchGame);
      }

      // 3. Simulation
      double reward = _playOut(searchGame);

      // 4. Backpropagation
      _backup(path, reward);
    }
    return _emitPv();
  }

  Node getNode(Game game) {
    return Node(game.state.hash);
  }

  Edge _selectBestChild(Node node) {
    double bestScore = -double.infinity;
    Edge? bestEdge;
    for (Edge edge in node.children) {
      final score = _calculatePuctScore(node, edge, 1.4);
      if (score > bestScore) {
        bestScore = score;
        bestEdge = edge;
      }
    }
    return bestEdge!;
  }

  // Future<void> _expandNode(Node node, Game game) async {
  //   // Get all legal moves from the current game state
  //   final List<Move> legalMoves = game.generateLegalMoves();

  //   // Get the full algebraic move history from the game
  //   final List<String> history = game.moveHistoryAlgebraic;

  //   // Prepare input for the GPT model with the <start> token
  //   final List<int> inputTokens = [stoi['<start>']!];
  //   if (history.isNotEmpty) {
  //     // Add all historical moves to the input sequence
  //     inputTokens.addAll(history.map((move) => stoi[move]!));
  //   }

  //   // Pass the full sequence to the policy network
  //   // The encoderOutput must be a non-empty list of ValueVector,
  //   // typically representing the board state. If you don't use a separate
  //   // encoder, you can pass a list of dummy values.
  //   final List<ValueVector> logits = await policyNetwork.forward(inputTokens, [
  //     ValueVector([Value(0)])
  //   ]);

  //   // If the model's output is also a single token, it means it's
  //   // predicting the next move.
  //   final ValueVector probabilities = logits.last.softmax();

  //   for (Move move in legalMoves) {
  //     final String moveAlgebraic = game.toAlgebraic(move);
  //     final int? moveTokenId = stoi[moveAlgebraic];

  //     if (moveTokenId != null) {
  //       final double priorProb = probabilities.values[moveTokenId].data;
  //       Edge edge = Edge(move, priorProb);
  //       node.children.add(edge);
  //     }
  //   }
  // }
  Future<void> _expandNode(Node node, Game game) async {
    // Get all legal moves from the current game state
    final List<Move> legalMoves = game.generateLegalMoves();

    // Get the full algebraic move history from the game
    final List<String> history = game.moveHistoryAlgebraic;

    // Prepare input for the GPT model with the <start> token
    final List<int> inputTokens = [stoi['<start>']!];
    if (history.isNotEmpty) {
      inputTokens.addAll(history.map((move) => stoi[move]!));
    }

    // Create a dummy encoder output with the correct block size
    final List<ValueVector> dummyEncoderOutput = List.generate(
      16, // Use your model's blockSize here (16 in your gpt_model.dart)
      (_) => ValueVector(
          List.filled(32, Value(0.0))), // Use your model's embedSize (64)
    );

    final List<ValueVector> logits =
        await policyNetwork.forward(inputTokens, dummyEncoderOutput);
    final ValueVector probabilities = logits.last.softmax();

    for (Move move in legalMoves) {
      final String moveAlgebraic = game.toAlgebraic(move);
      final int? moveTokenId = stoi[moveAlgebraic];

      if (moveTokenId != null) {
        final double priorProb = probabilities.values[moveTokenId].data;
        Edge edge = Edge(move, priorProb);
        node.children.add(edge);
      }
    }
  }

  double _playOut(Game game) {
    Game playoutGame = Game.fromPgn(game.pgn());
    while (!playoutGame.gameOver) {
      Move? randomMove = playoutGame.getRandomMove();
      if (randomMove != null) {
        playoutGame.makeMove(randomMove, false);
      } else {
        break;
      }
    }

    if (playoutGame.result is WonGame) {
      WonGame result = playoutGame.result as WonGame;
      return result.winner == game.turn ? 1.0 : -1.0;
    } else {
      return 0.0;
    }
  }

  void _backup(List<Edge> path, double reward) {
    for (Edge edge in path.reversed) {
      edge.visits++;
      edge.actionValue += reward;
      reward = -reward;
    }
    rootNode.visitCount++;
  }

  EngineResult _emitPv() {
    if (rootNode.children.isEmpty) throw Exception('No moves to choose from');
    Edge bestEdge = rootNode.children.reduce((a, b) {
      return a.visits > b.visits ? a : b;
    });
    return EngineResult(
      move: bestEdge.move,
      eval: bestEdge.actionValue / bestEdge.visits,
      depth: bestEdge.visits,
    );
  }

  double _calculatePuctScore(Node node, Edge edge, double c_puct) {
    final q_value = edge.visits == 0 ? 0.0 : edge.actionValue / edge.visits;
    // The new PUCT formula incorporates the prior probability from the policy network
    final explorationTerm = c_puct *
        edge.priorProbability *
        (sqrt(node.visitCount) / (1 + edge.visits));
    return q_value + explorationTerm;
  }
}
