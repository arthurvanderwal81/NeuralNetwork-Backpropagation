using System.Collections.Generic;

using NeuralNetLib.ActivationFunctions;

namespace NeuralNetLib.Layers
{   public class FullyConnectedLayer : AbstractLayer
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
                _neurons.Add(new Neuron(this));
            }

            _activationFunction = activationFunction;
        }

        public void CalculateOutput()
        {
            List<float> result = new List<float>();

            foreach (Neuron neuron in _neurons)
            {
                result.Add(neuron.CalculateOutput());
            }

            if (_activationFunction == null)
            {
                return;
            }

            IList<float> activatedResult = _activationFunction.Calculate(result);

            for (int i = 0; i < _neurons.Count; i++)
            {
                Neuron neuron = _neurons[i];

                neuron.OutputValue = activatedResult[i];
            }
        }

        public List<float> GetWeights()
        {
            List<float> result = new List<float>();

            foreach (Neuron neuron in _neurons)
            {
                result.AddRange(neuron.Weights);
            }

            return result;
        }
    }
}
