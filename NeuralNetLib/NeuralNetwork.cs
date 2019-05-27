using System;
using System.Linq;
using System.Diagnostics;
using System.Collections.Generic;

using NeuralNetLib.ActivationFunctions;
using NeuralNetLib.Layers;

namespace NeuralNetLib
{
    public class NeuralNetwork
    {
        private float _learningRate = 0.01f; //0.1f;
        private static Random _randomGenerator = new Random();
        private List<FullyConnectedLayer> _layers;

        private NeuralNetwork()
        {
            _layers = new List<FullyConnectedLayer>();
        }

        private static FullyConnectedLayer CreateLayer(int layerIndex, int neurons, AbstractActivationFunction activationFunction)
        {
            FullyConnectedLayer layer = new FullyConnectedLayer(layerIndex, neurons, activationFunction);

            return layer;
        }

        private static float GetRandomFloat(float lowerRange, float upperRange)
        {
            float range = upperRange - lowerRange;

            return (float)(_randomGenerator.NextDouble() * range + lowerRange);
        }

        private static void FullyConnect(Neuron neuron, FullyConnectedLayer layer)
        {
            for (int i = 0; i < layer.Count; i++)
            {
                neuron.OutputNeurons.Add(layer.Neurons[i]);

                layer.Neurons[i].InputNeurons.Add(neuron);
                layer.Neurons[i].Weights.Add(NeuralNetwork.GetRandomFloat(-1.0f, 1.0f));
            }
        }

        public static NeuralNetwork Create(int inputNeurons, int outputNeurons, int hiddenLayers, int hiddenLayerNeurons)
        {
            NeuralNetwork result = new NeuralNetwork();

            FullyConnectedLayer previousLayer = null;

            result._layers.Add(NeuralNetwork.CreateLayer(0, inputNeurons, null));

            for (int i = 0; i < hiddenLayers; i++)
            {
                result._layers.Add(NeuralNetwork.CreateLayer(i + 1, hiddenLayerNeurons, new ReLuActivationFunction()));

                previousLayer = result._layers[i];

                for (int neuronIndex = 0; neuronIndex < previousLayer.Count; neuronIndex++)
                {
                    NeuralNetwork.FullyConnect(previousLayer.Neurons[neuronIndex], result._layers[i + 1]);
                }
            }

            result._layers.Add(NeuralNetwork.CreateLayer(hiddenLayers, outputNeurons, new SigmoidActivationFunctions()));

            previousLayer = result._layers[result._layers.Count - 2];

            for (int neuronIndex = 0; neuronIndex < previousLayer.Count; neuronIndex++)
            {
                NeuralNetwork.FullyConnect(previousLayer.Neurons[neuronIndex], result._layers[result._layers.Count - 1]);
            }

            return result;
        }

        public static NeuralNetwork Create(float[][][] weights)
        {
            int inputNeurons = weights[0][0].Length;
            int outputNeurons = weights[weights.Length - 1].Length;
            int hiddenLayers = weights.Length - 1;
            int hiddenLayerNeurons = 0;

            if (hiddenLayers != 0)
            {
                hiddenLayerNeurons = weights[0].Length;
            }

            NeuralNetwork result = NeuralNetwork.Create(inputNeurons, outputNeurons, hiddenLayers, hiddenLayerNeurons);

            for (int weightLayerIndex = 0; weightLayerIndex < weights.Length; weightLayerIndex++)
            {
                int layerIndex = weightLayerIndex + 1;

                for (int neuronIndex = 0; neuronIndex < weights[weightLayerIndex].Length; neuronIndex++)
                {
                    result._layers[layerIndex].Neurons[neuronIndex].Weights.Clear();
                    result._layers[layerIndex].Neurons[neuronIndex].Weights.AddRange(weights[weightLayerIndex][neuronIndex]);
                }
            }

            return result;
        }

        public List<List<float>> GetWeights()
        {
            List<List<float>> result = new List<List<float>>();

            foreach (FullyConnectedLayer layer in _layers)
            {
                result.Add(layer.GetWeights());
            }

            return result;
        }

        public List<float> FeedForward(float[] input)
        {
            FullyConnectedLayer inputLayer = _layers[0];

            Debug.Assert(inputLayer.Count == input.Length);

            for (int neuronIndex = 0; neuronIndex < inputLayer.Count; neuronIndex++)
            {
                inputLayer.Neurons[neuronIndex].OutputValue = input[neuronIndex];
            }

            for (int layerIndex = 1; layerIndex < _layers.Count; layerIndex++)
            {
                _layers[layerIndex].CalculateOutput();
            }

            List<float> result = new List<float>();

            foreach (Neuron neuron in _layers[_layers.Count - 1].Neurons)
            {
                result.Add(neuron.OutputValue);
            }

            return result;
        }

        public float GetError(float[] expectedOutput)
        {
            float error = 0.0f;
            FullyConnectedLayer outputLayer = _layers[_layers.Count - 1];

            Debug.Assert(outputLayer.Count == expectedOutput.Length);

            for (int neuronIndex = 0; neuronIndex < outputLayer.Count; neuronIndex++)
            {
                float diff = outputLayer.Neurons[neuronIndex].OutputValue - expectedOutput[neuronIndex];
                diff *= diff;

                error += diff;
            }

            return (float)Math.Sqrt(error);
        }

        public float GetErrorClassification(float[] expectedOutput)
        {
            int maxExpectedOutputIndex = -1;
            float maxExpectedOutputValue = float.MinValue;

            int maxOutputIndex = -1;
            float maxOutputValue = float.MinValue;

            FullyConnectedLayer outputLayer = _layers[_layers.Count - 1];

            Debug.Assert(outputLayer.Count == expectedOutput.Length);

            for (int neuronIndex = 0; neuronIndex < outputLayer.Count; neuronIndex++)
            {
                if (expectedOutput[neuronIndex] > maxExpectedOutputValue)
                {
                    maxExpectedOutputValue = expectedOutput[neuronIndex];
                    maxExpectedOutputIndex = neuronIndex;
                }

                if (outputLayer.Neurons[neuronIndex].OutputValue > maxOutputValue)
                {
                    maxOutputValue = outputLayer.Neurons[neuronIndex].OutputValue;
                    maxOutputIndex = neuronIndex;
                }
            }

            return maxOutputIndex == maxExpectedOutputIndex ? 0.0f : 1.0f;
        }

        public void BackPropagate(float[] expectedOutput)
        {
            FullyConnectedLayer outputLayer = _layers[_layers.Count- 1];

            for (int neuronIndex = 0; neuronIndex < outputLayer.Count; neuronIndex++)
            {
                Neuron neuron = outputLayer.Neurons[neuronIndex];

                neuron.ErrorSignal = (expectedOutput[neuronIndex] - neuron.OutputValue) * outputLayer.ActivationFunction.Derivative(new float[] { neuron.OutputValue })[0];
            }

            for (int layerIndex = _layers.Count - 2; layerIndex > 0; layerIndex--)
            {
                FullyConnectedLayer layer = _layers[layerIndex];

                for (int neuronIndex = 0; neuronIndex < _layers[layerIndex].Count; neuronIndex++)
                {
                    Neuron neuron = layer.Neurons[neuronIndex];

                    neuron.ErrorSignal = 0.0f;

                    for (int outputNeuronIndex = 0; outputNeuronIndex < neuron.OutputNeurons.Count; outputNeuronIndex++)
                    {
                        Neuron outputNeuron = neuron.OutputNeurons[outputNeuronIndex];

                        neuron.ErrorSignal += outputNeuron.ErrorSignal * outputNeuron.Weights[neuronIndex] * layer.ActivationFunction.Derivative(new float[] { neuron.OutputValue })[0];
                    }
                }
            }

            // for all neurons
            // weight = weight + learning_rate * error * input

            for (int layerIndex = 1; layerIndex < _layers.Count; layerIndex++)
            {
                FullyConnectedLayer layer = _layers[layerIndex];

                for (int neuronIndex = 0; neuronIndex < layer.Count; neuronIndex++)
                {
                    Neuron neuron = layer.Neurons[neuronIndex];

                    for (int weightIndex = 0; weightIndex < neuron.Weights.Count; weightIndex++)
                    {
                        neuron.Weights[weightIndex] += _learningRate * neuron.ErrorSignal * neuron.InputNeurons[weightIndex].OutputValue;
                        /*
                        if (float.IsInfinity(neuron.Weights[weightIndex]))
                        {
                            throw new ArithmeticException();
                        }

                        if (neuron.Weights[weightIndex] > 2.0f)
                        {
                            throw new ArithmeticException();
                        }*/
                    }

                    // https://stackoverflow.com/questions/3775032/how-to-update-the-bias-in-neural-network-backpropagation
                    // Backprop bias - init bias to 0.0f
                    neuron.Bias += _learningRate * neuron.ErrorSignal * 1.0f;
                }
            }
        }
    }
}
