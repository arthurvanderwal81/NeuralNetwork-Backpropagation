using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetLib.Layers
{
    public class MaxPoolingLayer : AbstractLayer
    {
        private int _layerIndex;
        private AbstractLayer _previousLayer;
        private int[] _stride;
        private int[] _size; // size of pooling kernel

        public List<float[,]> OutputValues { get; set; }

        public MaxPoolingLayer(int layerIndex, AbstractLayer previousLayer)
        {
            _layerIndex = layerIndex;
            _previousLayer = previousLayer;

            _stride = new int[] { 2, 2 };
            _size = new int[] { 2, 2 };
        }

        public void CalculateOutput()
        {
            List<float[,]> result = new List<float[,]>();

            ConvolutionalLayer2D layer = _previousLayer as ConvolutionalLayer2D;

            List<float[,]> inputValues = layer.OutputValues;

            for (int kernelIndex = 0; kernelIndex < inputValues.Count; kernelIndex++)
            {
                int outputHeight = (inputValues[kernelIndex].GetLength(0) - _size[0]) / _stride[0] + 1;
                int outputWidth = (inputValues[kernelIndex].GetLength(1) - _size[1]) / _stride[1] + 1;

                float[,] kernelOutput = new float[outputHeight, outputWidth];

                for (int y = 0; y < kernelOutput.GetLength(0); y++)
                {
                    for (int x = 0; x < kernelOutput.GetLength(1); x++)
                    {
                        float[] kernelValues = new float[]
                        {
                            inputValues[kernelIndex][y * _stride[0], x * _stride[1]],
                            inputValues[kernelIndex][y * _stride[0], x * _stride[1] + 1],
                            inputValues[kernelIndex][y * _stride[0] + 1, x * _stride[1]],
                            inputValues[kernelIndex][y * _stride[0] + 1, x * _stride[1] + 1],
                        };

                        kernelOutput[y, x] = Enumerable.Max(kernelValues);
                    }
                }

                result.Add(kernelOutput);
            }

            OutputValues = result;
        }
    }
}
