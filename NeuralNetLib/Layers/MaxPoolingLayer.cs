using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetLib.Layers
{
    public class MaxPoolingLayer : AbstractLayer
    {
        private int _layerIndex;
        //private AbstractLayer _previousLayer;
        private int[] _stride;
        private int[] _size; // size of pooling kernel

        public List<float[,]> OutputValues { get; set; }

        public MaxPoolingLayer(int layerIndex)
        {
            _layerIndex = layerIndex;
            //_previousLayer = previousLayer;

            _stride = new int[] { 2, 2 };
            _size = new int[] { 2, 2 };
        }

        public override Array CalculateOutput(Array input)
        {
            float[,,] inputValues = input as float[,,];

            int outputHeight = (inputValues.GetLength(1) - _size[0]) / _stride[0] + 1;
            int outputWidth = (inputValues.GetLength(2) - _size[1]) / _stride[1] + 1;

            float[,,] result = new float[inputValues.GetLength(0), outputHeight, outputWidth];

            //ConvolutionalLayer2D layer = _previousLayer as ConvolutionalLayer2D;

            for (int kernelIndex = 0; kernelIndex < inputValues.GetLength(0); kernelIndex++)
            {
                //float[,] kernelOutput = new float[outputHeight, outputWidth];

                for (int y = 0; y < outputHeight; y++)
                {
                    for (int x = 0; x < outputWidth; x++)
                    {
                        float[] kernelValues = new float[]
                        {
                            inputValues[kernelIndex, y * _stride[0], x * _stride[1]],
                            inputValues[kernelIndex, y * _stride[0], x * _stride[1] + 1],
                            inputValues[kernelIndex, y * _stride[0] + 1, x * _stride[1]],
                            inputValues[kernelIndex, y * _stride[0] + 1, x * _stride[1] + 1],
                        };

                        result[kernelIndex, y, x] = Enumerable.Max(kernelValues);
                    }
                }

                //result.Add(kernelOutput);
            }

            return result;
        }

        public override Array BackPropagate(Array error, float learningRate)
        {
            return error;
        }
    }
}
