using System;
using System.Collections.Generic;

using NeuralNetLib.ActivationFunctions;

namespace NeuralNetLib.Layers
{
    public class FullyConnectedLayer : AbstractLayer
    {
        private int _layerIndex;
        private List<Neuron> _neurons;
        private AbstractActivationFunction _activationFunction;

        public int Count
        {
            get
            {
                return _neurons.Count;
            }
        }

        public List<Neuron> Neurons
        {
            get
            {
                return _neurons;
            }
        }

        public AbstractActivationFunction ActivationFunction
        {
            get
            {
                return _activationFunction;
            }
        }

        public FullyConnectedLayer(int layerIndex, int neurons, AbstractActivationFunction activationFunction)
        {
            _layerIndex = layerIndex;
            _neurons = new List<Neuron>();

            for (int i = 0; i < neurons; i++)
            {
                _neurons.Add(new Neuron());
            }

            _activationFunction = activationFunction;
        }

        public override Array CalculateOutput(Array input)
        {
            List<float> result = new List<float>();

            foreach (Neuron neuron in _neurons)
            {
                result.Add(neuron.CalculateOutput(input));
            }

            if (_activationFunction == null)
            {
                return result.ToArray();
            }

            float[] activatedResult = _activationFunction.Calculate(result.ToArray());

            // TODO Store activated result back in neurons or in layer

            return activatedResult;

            for (int i = 0; i < _neurons.Count; i++)
            {
                Neuron neuron = _neurons[i];

                neuron.OutputValue = activatedResult[i];
            }

            return result.ToArray();
        }

        public override Array BackPropagate(Array error, float learningRate)
        {
            return error;
        }

        /*
        public List<float> GetWeights()
        {
            List<float> result = new List<float>();

            foreach (Neuron neuron in _neurons)
            {
                result.AddRange(neuron.Weights);
            }

            return result;
        }
        */
    }
}
