import 'dart:math';

import 'package:bishop/bishop.dart';
import 'package:dart_torch/nn/value.dart';
import 'package:dart_torch/nn/value_vector.dart';
import 'dart:math';

class Engine {
  final Game game;

  Engine({required this.game});

  Future<EngineResult> search({
    int maxDepth = 50,
    int timeLimit = 5000,
    int timeBuffer = 2000,
    int debug = 0,
    int printBest = 0,
  }) async {
    if (game.gameOver) {
      print(game.drawn ? 'Draw' : 'Checkmate');
      return const EngineResult();
    }
    int endTime = DateTime.now().millisecondsSinceEpoch + timeLimit;
    int endBuffer = endTime + timeBuffer;
    int depthSearched = 0;
    List<Move> moves = game.generateLegalMoves();
    Map<Move, int> evals = {};
    for (Move m in moves) {
      evals[m] = 0;
    }
    for (int depth = 1; depth < maxDepth; depth++) {
      if (debug > 0) print('----- DEPTH $depth -----');
      for (Move m in moves) {
        game.makeMove(m, false);
        int eval = -negamax(
          depth,
          -Bishop.mateUpper,
          Bishop.mateUpper,
          game.turn.opponent,
          debug,
        );
        game.undo();
        evals[m] = eval;
        int now = DateTime.now().millisecondsSinceEpoch;
        if (now >= endBuffer) break;
      }
      moves.sort((a, b) => evals[b]!.compareTo(evals[a]!));
      depthSearched = depth;
      int now = DateTime.now().millisecondsSinceEpoch;
      if (now >= endTime) break;
    }
    if (printBest > 0) {
      print('-- Best Moves --');
      for (Move m in moves.take(printBest)) {
        print('${game.toSan(m)}: ${evals[m]}');
      }
    }

    return EngineResult(
      move: moves.first,
      eval: evals[moves.first]!.toDouble(),
      depth: depthSearched,
    );
  }

  int negamax(int depth, int alpha, int beta, Colour player, [int debug = 0]) {
    int value = -Bishop.mateUpper;
    if (game.drawn) return 0;
    final result = game.result;
    if (result is WonGame) {
      return result.winner == player ? -Bishop.mateUpper : Bishop.mateUpper;
    }
    List<Move> moves = game.generateLegalMoves();
    if (moves.isEmpty) {
      if (game.turn == player) {
        return Bishop.mateUpper;
      } else {
        return -Bishop.mateUpper;
      }
    }
    // -evaluate because we are currently looking at this asthe other player
    if (depth == 0) return -game.evaluate(player);
    if (moves.isNotEmpty) {
      int a = alpha;
      for (Move m in moves) {
        game.makeMove(m, false);
        int v = -negamax(depth - 1, -beta, -a, player.opponent, debug - 1);
        game.undo();
        value = max(v, value);
        a = max(a, value);
        if (a >= beta) break;
      }
    }
    return value;
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
  // final Node? parent;
  // final int action; // The action that led to this node
  // BishopState state;
  int key;
  // ValueVector hiddenState;
  late PredictionOutput prediction; // Policy and Value from the model

  int visitCount = 0;
  double totalValue = 0.0;
  Value reward = Value(0.0);
  List<Edge> children = [];

  Node(
    this.key,
    // {required this.hiddenState,
  );

  double get qValue => visitCount == 0 ? 0.0 : totalValue / visitCount;

  // TODO: Add methods for PUCT score calculation, expansion, etc.
}

Map<int, Map<int, Node>> mctsTree = {};
final MUDULO = 4000;

Node getNode(Game game) {
  final key = game.state.hash;
  final entry = key % MUDULO;
  Map<int, Node>? slot = mctsTree[entry];

  if (slot != null) {
    for (int x = 0; x < slot.length; x++) {
      if (slot[x] != null && slot[x]!.key == key) {
        return slot[x]!;
      }
    }
    slot[key] = Node(key);
  }
  //  else {
  //   mctsTree[entry] = {key: Node(key)};
  // }

  mctsTree[entry] = {key: Node(key)};
  slot = mctsTree[entry];
  return slot![key]!;
}

class Mcts {
  int ply = 0;
  late Node rootNode;
  List<Node> nodes = [];
  List<Edge> edges = [];
  Game game;
  Mcts(this.game);

  EngineResult monteCarlo(Game pos) {
    pos = game;
    createRoot();

    for (int x = 0; x < 20; x++) {
      treePolicy();
      final reward = playOutPolicy();
      backup(reward);
    }
    return emitPv();
  }

  void createRoot() {
    nodes = [];
    edges = [];
    mctsTree = {};
    ply = 0;
    rootNode = getNode(game);
    nodes.add(rootNode);
  }

  bool isRoot() {
    return rootNode.key == currentNode().key;
  }

  void treePolicy() {
    print("Entering tree policy for ply: ply: $ply");
    // if (ply != 0) throw Exception('ply cannot be: $ply');
    while (currentNode().visitCount != 0) {
      Edge edge = bestClhild()!;
      // game.makeMove(edge!.move, false);
      edge.visits++;
      edge.reward = edge.reward;
      edge.actionValue = edge.actionValue;
      edge.meanActionValue = edge.actionValue / edge.visits;

      doMove(edge);
    }
  }

  double playOutPolicy() {
    print("Entering playou policy");
    if (game.drawn) return 0.0;
    final result = game.result;
    if (result is WonGame) {
      return Bishop.mateUpper.toDouble();
    }
    if (result is WonGame) {
      return Bishop.mateUpper.toDouble();
    }

    generateMoves(currentNode());

    if (currentNode().children.isEmpty) return -Bishop.mateUpper.toDouble();
    // print(currentNode().children);
    return currentNode().children.reduce((a, b) {
      if (a.reward > b.reward) {
        return a;
      } else {
        return b;
      }
    }).reward;
  }

  void backup(double reward) {
    print("Entering backup policy");
    while (!isRoot()) {
      // game.undo();
      if (edges.isEmpty) throw Exception("Edges cannot be empty for ply: $ply");
      undoMove();
      edges[ply].visits--;
      edges[ply].visits++;
      edges[ply].reward = reward;
      edges[ply].actionValue += reward;
      edges[ply].meanActionValue = edges[ply].actionValue / edges[ply].visits;
    }
    // if (ply != 0) throw Exception('ply cannot be: $ply');
  }

  EngineResult emitPv() {
    // return rootNode.children.first.move;
    // print('is root: ${isRoot()}');
    // print('is root: ${currentNode().key}');
    // if (ply != 0) throw Exception('ply cannot be: $ply');

    if (currentNode().children.isEmpty) throw Exception('ply cannot be: empty');
    // print('Children: ${currentNode().children}');
    final bestEdge = currentNode().children.reduce((a, b) {
      if (a.visits > b.visits) {
        return a;
      } else {
        return b;
      }
    });
    return EngineResult(
      move: bestEdge.move,
      eval: bestEdge.reward,
      depth: bestEdge.visits,
    );
  }

  Node currentNode() {
    // if (ply >= nodes.length) {
    //   nodes.add(getNode(game));
    //   // edges.add(edge);
    // }
    // print("Nodes: ${nodes}");
    return nodes[ply];
  }

  void doMove(Edge edge) {
    if (ply + 1 >= nodes.length) {
      nodes.add(getNode(game));
    }
    if (ply + 1 >= edges.length) {
      edges.add(edge);
    }
    edges[ply] = edge;
    game.makeMove(edge.move, false);
    ply++;
  }

  void undoMove() {
    // if (ply >= nodes.length) {
    //   nodes.add(getNode(game));
    //   // edges.add(edge);
    // }

    ply--;
    game.undo();
  }

  void generateMoves(Node node) {
    int value = -Bishop.mateUpper;
    List<Move> moves = game.generateLegalMoves();
    for (Move move in moves) {
      Edge edge = Edge(move);

      game.makeMove(move, false);
      // ply++;
      int v = -negamax(
          1, -(-Bishop.mateUpper), -(Bishop.mateUpper), game.turn.opponent);
      game.undo();
      // ply--;
      value = max(v, value);
      edge.reward = value.toDouble();
      node.children.add(edge);

      // final a = max(a, value);
      // if (a >= beta) break;
    }
    node.visitCount = 1;
  }

  int negamax(int depth, int alpha, int beta, Colour player, [int debug = 0]) {
    int value = -Bishop.mateUpper;
    if (game.drawn) return 0;
    final result = game.result;
    if (result is WonGame) {
      return result.winner == player ? -Bishop.mateUpper : Bishop.mateUpper;
    }
    List<Move> moves = game.generateLegalMoves();
    if (moves.isEmpty) {
      if (game.turn == player) {
        return Bishop.mateUpper;
      } else {
        return -Bishop.mateUpper;
      }
    }
    // -evaluate because we are currently looking at this asthe other player
    if (depth == 0) return -game.evaluate(player);
    if (moves.isNotEmpty) {
      int a = alpha;
      for (Move m in moves) {
        game.makeMove(m, false);
        int v = -negamax(depth - 1, -beta, -a, player.opponent, debug - 1);
        game.undo();
        value = max(v, value);
        a = max(a, value);
        if (a >= beta) break;
      }
    }
    return value;
  }

  Edge? bestClhild() {
    double bestScore = -Bishop.mateUpper.toDouble();
    Edge? bestEdge;
    for (Edge edge in currentNode().children) {
      final score = calculatePuctScore(currentNode(), edge, 0.7);
      if (score > bestScore) {
        bestScore = score;
        bestEdge = edge;
      }
    }
    return bestEdge;
  }

  double calculatePuctScore(Node node, Edge edge, double c_puct) {
    // Get required values from the node for the given action
    final q_value = edge.actionValue;
    final p_value = edge.reward;
    final n_value = edge.visits;
    final total_n = node.visitCount;

    // The exploitation term: Q(s,a)
    final exploitationTerm = q_value;

    // The exploration term: c_puct * P(s,a) * sqrt(sum(N(s,b))) / (1 + N(s,a))
    final explorationTerm = c_puct * p_value * sqrt(total_n) / (1 + n_value);

    return exploitationTerm + explorationTerm;
  }
}

// A container for the output of the prediction function (f)
class PredictionOutput {
  final ValueVector policyLogits;
  final Value value;
  PredictionOutput(this.policyLogits, this.value);
}
