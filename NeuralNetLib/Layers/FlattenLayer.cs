using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NeuralNetLib.ActivationFunctions;

namespace NeuralNetLib.Layers
{
    public class FlattenLayer : AbstractLayer
    {
        private int _layerIndex;
        private AbstractLayer _previousLayer;
        //private List<Neuron> _neurons;

        public List<float> OutputValues { get; set; }

        public FlattenLayer(int layerIndex, AbstractLayer previousLayer)
        {
            if (!(previousLayer is ConvolutionalLayer2D) && !(previousLayer is MaxPoolingLayer))
            {
                throw new Exception("Invalid previous layer for flatten layer");
            }

            _layerIndex = layerIndex;
            _previousLayer = previousLayer;

            //ConvolutionalLayer2D layer = previousLayer as ConvolutionalLayer2D;

            //layer.OutputValues

            //int neurons = previousLayer.sizeof output volume = kernels x kernel result dimensions
            /*
            for (int i = 0; i < neurons; i++)
            {
                _neurons.Add(new Neuron(this));
            }
            */
        }

        public void CalculateOutput()
        {
            ConvolutionalLayer2D layer = _previousLayer as ConvolutionalLayer2D;

            OutputValues = new List<float>();

            for (int kernelIndex = 0; kernelIndex < layer.OutputValues.Count; kernelIndex++)
            {
                for (int y = 0; y < layer.OutputValues[kernelIndex].GetLength(0); y++)
                {
                    for (int x = 0; x < layer.OutputValues[kernelIndex].GetLength(1); x++)
                    {
                        OutputValues.Add(layer.OutputValues[kernelIndex][y, x]);
                    }
                }
            }
        }
    }
}
