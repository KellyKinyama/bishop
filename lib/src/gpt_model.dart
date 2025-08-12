// gpt_model.dart

// import 'dart:math';
// import 'package:bishop/bishop.dart';
import 'package:dart_torch/transformer/transformer_decoder.dart';
import 'package:dart_torch/nn/value.dart';
// import 'transformer_decoder.dart';
// import 'value.dart';
// import 'value_vector.dart';

// This is a re-implementation of your SGD optimizer, assuming it's a dependency.
class SGD {
  final List<Value> parameters;
  final double learningRate;

  SGD(this.parameters, this.learningRate);

  void step() {
    for (final p in parameters) {
      if (p.grad != null) {
        p.data -= learningRate * p.grad;
      }
    }
  }

  void zeroGrad() {
    for (final p in parameters) {
      p.grad = 0;
    }
  }
}

// Global helper function to generate a vocabulary from a dataset of games.
Map<String, int> _generateChessVocab(List<List<String>> gameSequences) {
  final Map<String, int> stoi = {};
  int idCounter = 0;

  // Add special tokens
  stoi['<start>'] = idCounter++;
  stoi['<end>'] = idCounter++;
  stoi['<pad>'] = idCounter++;

  // Populate vocabulary with all unique moves from the training data
  for (final game in gameSequences) {
    for (final move in game) {
      if (!stoi.containsKey(move)) {
        stoi[move] = idCounter++;
      }
    }
  }

  return stoi;
}

// This function will be called from your uci.dart file to get the initialized model and vocabulary.
(TransformerDecoder, Map<String, int>, Map<int, String>) createGptModel() {
  // A small dataset of complete games for demonstration
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
    // Add more games here to improve the model's performance
  ];

  final Map<String, int> stoi = _generateChessVocab(rawGameSequences);
  final Map<int, String> itos = stoi.map((key, value) => MapEntry(value, key));
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

  // Here you would load a pre-trained model state if available.
  // For this example, we assume the model is untrained.

  return (gptModel, stoi, itos);
}
