// gpt_model.dart
import 'package:dart_torch/transformer/transformer_decoder.dart';
import 'package:dart_torch/nn/value.dart';
import 'package:dart_torch/nn/value_vector.dart';
// ... (imports and existing classes like SGD)

class SGD {
  final List<Value> parameters;
  final double learningRate;

  SGD(this.parameters, this.learningRate);

  void step() {
    for (final p in parameters) {
      if (p.grad != null) {
        p.data -= learningRate * p.grad!;
      }
    }
  }

  void zeroGrad() {
    for (final p in parameters) {
      p.grad = 0;
    }
  }
}

class GptModel {
  final TransformerDecoder decoder;
  final SGD optimizer;
  final Map<String, int> stoi;
  final Map<int, String> itos;
  final int vocabSize;
  final int startTokenId;
  final int padTokenId;
  final int endTokenId;
  final int blockSize;

  GptModel(
    this.decoder,
    this.stoi,
    this.itos, {
    required this.vocabSize,
    required this.startTokenId,
    required this.padTokenId,
    required this.endTokenId,
    required this.blockSize,
  }) : optimizer = SGD(decoder.parameters(), 0.01);

  static Map<String, int> _generateChessVocab(
      List<List<String>> gameSequences) {
    final Map<String, int> stoi = {};
    int idCounter = 0;
    stoi['<start>'] = idCounter++;
    stoi['<end>'] = idCounter++;
    stoi['<pad>'] = idCounter++;
    for (final game in gameSequences) {
      for (final move in game) {
        if (!stoi.containsKey(move)) {
          stoi[move] = idCounter++;
        }
      }
    }
    return stoi;
  }

  static GptModel create() {
    final List<List<String>> rawGameSequences = [
      [
        'e2e4',
        'e7e5',
        'g1f3',
        'b8c6',
        'f1c4',
        'f8c5',
        'c2c3',
        'g8f6',
        'd2d4',
        'e5d4',
        'c3d4',
        'c4b4'
      ],
      [
        'd2d4',
        'd7d5',
        'c2c4',
        'e7e6',
        'c1f4',
        'g8f6',
        'g1f3',
        'c7c5',
        'c4d5',
        'f6d5',
        'e3e4',
        'd5f4',
        'd1d8'
      ],
    ];
    final Map<String, int> stoi = _generateChessVocab(rawGameSequences);
    final Map<int, String> itos =
        stoi.map((key, value) => MapEntry(value, key));
    final int vocabSize = stoi.length;
    const int embedSize = 32;
    const int blockSize = 16;
    const int numLayers = 2;
    const int numHeads = 4;
    final gptModel = TransformerDecoder(
      vocabSize: vocabSize,
      embedSize: embedSize,
      blockSize: blockSize,
      numLayers: numLayers,
      numHeads: numHeads,
      encoderEmbedSize: embedSize,
    );

    return GptModel(
      gptModel,
      stoi,
      itos,
      vocabSize: vocabSize,
      startTokenId: stoi['<start>']!,
      padTokenId: stoi['<pad>']!,
      endTokenId: stoi['<end>']!,
      blockSize: blockSize,
    );
  }

  Future<void> trainOnSequence(
      List<String> gameMoves, double gameResult,) async {
    final List<int> tokenizedSeq = [
      startTokenId,
      ...gameMoves.map((move) => stoi[move]!),
      endTokenId,
    ];

    List<int> input = tokenizedSeq.sublist(0, tokenizedSeq.length - 1);
    List<int> target = tokenizedSeq.sublist(1);

      if (input.length > blockSize) {
      input = input.sublist(0, blockSize);
      target = target.sublist(0, blockSize);
    }


    optimizer.zeroGrad();

    // Call the model and get both the policy and value outputs
    final List<ValueVector> outputs = decoder.forward(input, []);
    final ValueVector policyLogits = outputs[0];
    final ValueVector valueOutput = outputs[1];

    // 1. Calculate Policy Loss (Cross-Entropy)
    Value policyLoss = Value(0.0);
    int activeTokens = 0;
    for (int t = 0; t < policyLogits.values.length; t++) {
      final ValueVector tokenLogits = policyLogits; // Assuming single output
      final int trueTargetId = target[t];
      final Value trueLogit = tokenLogits.values[trueTargetId];
      final Value sumExpLogits =
          tokenLogits.values.map((v) => v.exp()).reduce((a, b) => a + b);
      final Value logSumExp = sumExpLogits.log();
      final Value negLogProb = logSumExp - trueLogit;
      policyLoss += negLogProb;
      activeTokens++;
    }

    if (activeTokens > 0) {
      policyLoss = policyLoss / Value(activeTokens.toDouble());
    }

    // 2. Calculate Value Loss (Mean Squared Error)
    // The target value is the final game result (-1 for loss, 0 for draw, 1 for win)
    final Value predictedValue = valueOutput.values.first;
    final Value targetValue = Value(gameResult);
    final Value valueLoss = (predictedValue - targetValue).pow(2);

    // 3. Combine losses
    // You can use a weight to balance the two losses, e.g., 0.5 for each
    final Value totalLoss = policyLoss + valueLoss;

    totalLoss.backward();
    optimizer.step();

    print(
        'Trained on ${gameMoves.length} moves. Total Loss: ${totalLoss.data}');
  }
}
