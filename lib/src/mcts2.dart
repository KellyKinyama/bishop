import 'dart:math';

import 'package:bishop/bishop.dart';

class Mcts {
  Node rootNode;
  Game game;

  Mcts(this.game) : rootNode = Node(game.state.hash);

  EngineResult monteCarlo(Game pos) {
    // Start with a new game clone for each MCTS run
    rootNode = Node(pos.state.hash);

    for (int i = 0; i < 200; i++) {
      // 1. Selection
      List<Edge> path = [];
      Node currentNode = rootNode;
      Game searchGame = pos; // Clone the game for this iteration

      while (!searchGame.gameOver && currentNode.children.isNotEmpty) {
        Edge bestEdge = _selectBestChild(currentNode);
        path.add(bestEdge);
        searchGame.makeMove(bestEdge.move, false);
        currentNode = getNode(searchGame);
      }

      // 2. Expansion
      if (!searchGame.gameOver) {
        _expandNode(currentNode, searchGame);
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

  void _expandNode(Node node, Game game) {
    List<Move> moves = game.generateLegalMoves();
    for (Move move in moves) {
      Edge edge = Edge(move);
      node.children.add(edge);
    }
  }

  double _playOut(Game game) {
    Game playoutGame = game;
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
    final explorationTerm = c_puct * sqrt(node.visitCount) / (1 + edge.visits);
    return q_value + explorationTerm;
  }
}

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
  Edge(this.move);
}

class Node {
  int key;
  int visitCount = 0;
  double totalValue = 0.0;
  List<Edge> children = [];
  Node(this.key);
  double get qValue => visitCount == 0 ? 0.0 : totalValue / visitCount;
}
