// mcts3.dart

import 'dart:math';
import 'package:bishop/bishop.dart';
import 'package:dart_torch/nn/value.dart';
import 'package:dart_torch/nn/value_vector.dart';
// import 'gpt_model.dart';
import 'package:dart_torch/transformer/transformer_decoder.dart';

class Mcts {
  Node rootNode;
  final TransformerDecoder policyNetwork;
  final Map<String, int> stoi;
  final Map<int, String> itos;
  final Game game;

  Mcts(this.game, this.policyNetwork, this.stoi, this.itos)
      : rootNode = Node(game.state.hash);

  Future<EngineResult> monteCarlo({
    int maxIterations = 5000,
    int timeLimit = 5000,
  }) async {
    final int endTime = DateTime.now().millisecondsSinceEpoch + timeLimit;

    rootNode = Node(game.state.hash);

    for (int i = 0; i < maxIterations; i++) {
      if (DateTime.now().millisecondsSinceEpoch > endTime) {
        print('MCTS search timed out after $i iterations.');
        break;
      }

      // 1. Selection
      List<Edge> path = [];
      Node currentNode = rootNode;
      Game searchGame = Game.fromPgn(game.pgn());

      while (!searchGame.gameOver && currentNode.children.isNotEmpty) {
        Edge bestEdge = _selectBestChild(currentNode, searchGame);
        path.add(bestEdge);
        searchGame.makeMove(bestEdge.move, false);
        currentNode = _getNode(searchGame, currentNode, bestEdge.move);
      }

      // 2. Expansion
      if (!searchGame.gameOver) {
        await _expandNode(currentNode, searchGame);
      }

      // 3. Simulation
      final double simulationResult = _simulate(searchGame);

      // 4. Backpropagation
      _backpropagate(path, simulationResult);
    }

    if (rootNode.children.isEmpty) {
      return const EngineResult();
    }

    Edge bestEdge =
        rootNode.children.reduce((a, b) => a.visits > b.visits ? a : b);

    return EngineResult(
      move: bestEdge.move,
      eval: bestEdge.actionValue,
      depth: bestEdge.visits,
    );
  }

  Node _getNode(Game searchGame, Node parentNode, Move move) {
    final hash = searchGame.state.hash;
    final existingEdge = parentNode.children.firstWhere(
      (edge) => edge.move == move,
      orElse: () => throw Exception('Node for move not found'),
    );
    if (existingEdge.childNode == null) {
      existingEdge.childNode = Node(hash);
    }
    return existingEdge.childNode!;
  }

  Edge _selectBestChild(Node node, Game game) {
    double bestScore = -double.infinity;
    Edge? bestEdge;

    // The score is based on the current player. If the Q-value is positive,
    // it's good for the player who just moved (the parent node's player).
    // The child node's player wants to minimize this value (maximize their own score).
    for (Edge edge in node.children) {
      final qValue = edge.visits == 0 ? 0.0 : edge.actionValue / edge.visits;
      final score = qValue + _calculatePuctScore(node, edge, 1.0);
      if (score > bestScore) {
        bestScore = score;
        bestEdge = edge;
      }
    }
    return bestEdge!;
  }

  Future<void> _expandNode(Node node, Game game) async {
    final List<Move> legalMoves = game.generateLegalMoves();
    final List<String> history = game.moveHistoryAlgebraic;

    final List<int> inputTokens = [stoi['<start>']!];
    if (history.isNotEmpty) {
      inputTokens.addAll(history.map((move) => stoi[move]!));
    }

    final List<ValueVector> dummyEncoderOutput = List.generate(
      16,
      (_) => ValueVector(List.filled(64, Value(0.0))),
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

  double _simulate(Game game) {
    Game rolloutGame = Game.fromPgn(game.pgn());
    while (!rolloutGame.gameOver) {
      final legalMoves = rolloutGame.generateLegalMoves();
      if (legalMoves.isEmpty) break;
      final randomMove = legalMoves[Random().nextInt(legalMoves.length)];
      rolloutGame.makeMove(randomMove);
    }

    final int? winner = rolloutGame.winner;
    if (winner == game.turn) {
      return 1.0;
    } else if (winner == null) {
      return 0.0;
    } else {
      return -1.0;
    }
  }

  void _backpropagate(List<Edge> path, double reward) {
    for (Edge edge in path.reversed) {
      edge.visits++;
      edge.actionValue += reward;
      reward = -reward; // Invert the reward for the previous player
    }
    rootNode.visitCount++;
  }

  double _calculatePuctScore(Node node, Edge edge, double c_puct) {
    final q_value = edge.visits == 0 ? 0.0 : edge.actionValue / edge.visits;
    final explorationTerm =
        c_puct * edge.reward * sqrt(node.visitCount) / (1 + edge.visits);
    return q_value + explorationTerm;
  }
}

class Node {
  final int hash;
  int visitCount;
  List<Edge> children;

  Node(this.hash)
      : visitCount = 0,
        children = [];
}

class Edge {
  final Move move;
  double reward;
  double actionValue;
  int visits;
  Node? childNode;

  Edge(this.move, this.reward)
      : actionValue = 0.0,
        visits = 0;
}

class EngineResult {
  final Move? move;
  final double? eval;
  final int? depth;
  bool get hasMove => move != null;
  const EngineResult({this.move, this.eval, this.depth});
}
