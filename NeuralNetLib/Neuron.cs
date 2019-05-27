using System;
using System.Collections.Generic;

using NeuralNetLib.Layers;

namespace NeuralNetLib
{
    public class Neuron
    {
        private List<float> _weights;
        private List<float> _accumulatedWeights;
        private float _bias;

        private FullyConnectedLayer _layer;

        private List<Neuron> _inputNeurons;
        private List<Neuron> _outputNeurons;

        //private float _outputValue;

        public List<float> Weights
        {
            get
            {
                return _weights;
            }
        }

        public List<float> AccumulatedWeights
        {
            get
            {
                return _accumulatedWeights;
            }
        }

        public float Bias
        {
            get
            {
                return _bias;
            }
            set
            {
                _bias = value;
            }
        }

        public List<Neuron> InputNeurons
        {
            get
            {
                return _inputNeurons;
            }
        }
        public List<Neuron> OutputNeurons
        {
            get
            {
                return _outputNeurons;
            }
        }

        public float OutputValue { get; set; }
        public float ErrorSignal { get; set; }

        public Neuron(FullyConnectedLayer layer)
        {
            _layer = layer;

            _inputNeurons = new List<Neuron>();
            _outputNeurons = new List<Neuron>();

            _weights = new List<float>();
            _accumulatedWeights = new List<float>();

            _bias = 0.0f;
        }

        public float CalculateOutput()
        {
            OutputValue = 0.0f;

            for (int i = 0; i < _inputNeurons.Count; i++)
            {
                Neuron neuron = _inputNeurons[i];
                float weight = _weights[i];

                OutputValue += weight * neuron.OutputValue;
            }

            OutputValue += _bias;

            return OutputValue;
        }
    }
}
