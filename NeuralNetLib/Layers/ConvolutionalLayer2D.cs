using System;
using System.Collections.Generic;

using NeuralNetLib.ActivationFunctions;

namespace NeuralNetLib.Layers
{
    public class ConvolutionalLayer2D : AbstractLayer
    {
        public enum Padding
        {
            Same, // pad with 0 to make input dimension same as output dimensions
            None
        }

        private int _layerIndex;
        public List<Kernel> _kernels;

        /// <summary>
        /// y, x
        /// </summary>
        private int[] _strides;
        private Padding _padding;
        private AbstractActivationFunction _activationFunction;

        /// <summary>
        /// [channels, y, x]
        /// </summary>
        public float[,,] InputValues { get; set; }
        public List<float[,]> OutputValues { get; set; }

        public AbstractActivationFunction ActivationFunction
        {
            get
            {
                return _activationFunction;
            }
        }

        public ConvolutionalLayer2D(int layerIndex, int[] inputShape, int kernels, int[] kernelSize, int[] strides, Padding padding, AbstractActivationFunction activationFunction)
        { 
            _layerIndex = layerIndex;
            _strides = strides;
            _padding = padding;
            _kernels = new List<Kernel>();

            for (int i = 0; i < kernels; i++)
            {
                _kernels.Add(new Kernel(kernelSize));
            }

            _activationFunction = activationFunction;

            if (inputShape != null)
            {
                // This means it's the first conv2d layer
                InputValues = new float[inputShape[0], inputShape[1], inputShape[2]];
            }
        }

        public float[,,] ZeroPad(float[,,] inputValues, int kernelIndex)
        {           
            int totalPaddingX = (inputValues.GetLength(2) - 1) * _strides[1] - inputValues.GetLength(2) + _kernels[kernelIndex].Weights.GetLength(2);
            int totalPaddingY = (inputValues.GetLength(1) - 1) * _strides[0] - inputValues.GetLength(1) + _kernels[kernelIndex].Weights.GetLength(1);

            //totalPaddingX = 5;
            //totalPaddingY = 9;

            int prepadX = totalPaddingX / 2;
            int prepadY = totalPaddingY / 2;

            int postpadX = prepadX;
            int postpadY = prepadY;

            if ((totalPaddingX % 2) != 0)
            {
                postpadX = prepadX + 1;
            }

            if ((totalPaddingY % 2) != 0)
            {
                postpadY = prepadY + 1;
            }

            //Console.WriteLine(string.Format("Padding: ({0}, {1})", postpadX, postpadY));

            // result is initialized with 0.0f -> our padding value
            float[,,] result = new float[inputValues.GetLength(0), inputValues.GetLength(1) + totalPaddingY, inputValues.GetLength(2) + totalPaddingX];

            for (int channel = 0; channel < inputValues.GetLength(0); channel++)
            {
                for (int y = 0; y < inputValues.GetLength(1); y++)
                {
                    Buffer.BlockCopy(inputValues, ((channel * inputValues.GetLength(1) * inputValues.GetLength(2)) + (y * inputValues.GetLength(2))) * sizeof(float), 
                                     result,      ((channel * result.GetLength(1) * result.GetLength(2)) + (((y + prepadY) * result.GetLength(2) + prepadX))) * sizeof(float), inputValues.GetLength(2) * sizeof(float));
                }
            }

            /*
            for (int channel = 0; channel < result.GetLength(0); channel++)
            {
                for (int y = 0; y < result.GetLength(1); y++)
                {
                    for (int x = 0; x < result.GetLength(2); x++)
                    {
                        Console.Write(string.Format("{0} ", result[channel, y, x]));
                    }

                    Console.Write("\n");
                }

                Console.WriteLine("==========================================================================");
            }
            */

            return result;
        }

        public int GetOutputChannels(int kernelIndex)
        {
            return _kernels[kernelIndex].Weights.GetLength(0);
        }

        public int GetKernelCount()
        {
            return _kernels.Count;
        }

        public void CalculateOutput()
        {
            List<float[,]> result = new List<float[,]>();

            for (int kernelIndex = 0; kernelIndex < _kernels.Count; kernelIndex++)
            {
                float[,,] inputValues = InputValues;

                if (_padding == Padding.Same)
                {
                    inputValues = ZeroPad(inputValues, kernelIndex);
                }

                int outputHeight = (inputValues.GetLength(1) - _kernels[kernelIndex].Weights.GetLength(1)) / _strides[0] + 1;
                int outputWidth =  (inputValues.GetLength(2) - _kernels[kernelIndex].Weights.GetLength(2)) / _strides[1] + 1;

                float[,] kernelOutput = new float[outputHeight, outputWidth];

                for (int channel = 0; channel < inputValues.GetLength(0); channel++)
                {
                    /*
                    int resultY = 0;

                    for (int y = 0; y < (inputValues.GetLength(1) - _kernels[kernelIndex].Weights.GetLength(1)); y += _strides[0])
                    {
                        int resultX = 0;

                        for (int x = 0; x < (inputValues.GetLength(2) - _kernels[kernelIndex].Weights.GetLength(2)); x += _strides[1])
                        {
                            kernelOutput[resultY, resultX] = _kernels[kernelIndex].DotProduct(inputValues, x, y);

                            resultX++;
                        }

                        resultY++;
                    }
                    */

                    for (int y = 0; y < kernelOutput.GetLength(0); y++)
                    {
                        for (int x = 0; x < kernelOutput.GetLength(1); x++)
                        {
                            kernelOutput[y, x] = _kernels[kernelIndex].DotProduct(inputValues, x * _strides[1], y * _strides[0]);
                        }
                    }
                }

                result.Add(kernelOutput);
            }

            OutputValues = result;
        }
    }
}
