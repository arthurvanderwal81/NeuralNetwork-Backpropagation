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
        private int[] _inputShape;

        //private AbstractLayer _previousLayer;
        //private List<Neuron> _neurons;

        //public List<float> OutputValues { get; set; }

        public FlattenLayer(int layerIndex)
        {
            /*
            if (!(previousLayer is ConvolutionalLayer2D) && !(previousLayer is MaxPoolingLayer))
            {
                throw new Exception("Invalid previous layer for flatten layer");
            }*/

            _layerIndex = layerIndex;
            //_previousLayer = previousLayer;

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

        public override Array CalculateOutput(Array input)
        {
            float[,,] inputValues = input as float[,,];

            // this can be done once, in a 'compile' like step
            _inputShape = new int[] { inputValues.GetLength(0), inputValues.GetLength(1), inputValues.GetLength(2) };

            List<float> result = new List<float>();

            for (int kernelIndex = 0; kernelIndex < inputValues.GetLength(0); kernelIndex++)
            {
                for (int y = 0; y < inputValues.GetLength(1); y++)
                {
                    for (int x = 0; x < inputValues.GetLength(2); x++)
                    {
                        result.Add(inputValues[kernelIndex, y, x]);
                    }
                }
            }

            return result.ToArray();
        }

        public override Array BackPropagate(Array error, float learningRate)
        {
            float[] errorSignal = error as float[];
            int flattenedLayerIndex = 0;

            float[,,] result = new float[_inputShape[0], _inputShape[1], _inputShape[2]];

            for (int channel = 0; channel < _inputShape[0]; channel++)
            {
                for (int y = 0; y < _inputShape[1]; y++)
                {
                    for (int x = 0; x < _inputShape[2]; x++)
                    {
                        result[channel, y, x] = errorSignal[flattenedLayerIndex];

                        flattenedLayerIndex++;
                    }
                }
            }

            return result;
        }
    }
}
