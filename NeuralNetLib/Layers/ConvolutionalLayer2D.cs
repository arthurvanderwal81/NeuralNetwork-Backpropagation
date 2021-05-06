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

        private int[] _kernelSize;
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
        //public float[,,] InputValues { get; set; }
        public float[,,] OutputValues { get; set; }

        public AbstractActivationFunction ActivationFunction
        {
            get
            {
                return _activationFunction;
            }
        }

        public ConvolutionalLayer2D(int layerIndex, int kernels, int[] kernelSize, int[] strides, Padding padding, AbstractActivationFunction activationFunction)
        { 
            _layerIndex = layerIndex;

            _strides = strides;
            _padding = padding;

            _kernelSize = kernelSize;
            _kernels = new List<Kernel>();

            for (int i = 0; i < kernels; i++)
            {
                _kernels.Add(new Kernel(kernelSize));
            }

            _activationFunction = activationFunction;
        }

        public Array ZeroPad(Array inputValues)
        {           
            int totalPaddingX = (inputValues.GetLength(2) - 1) * _strides[1] - inputValues.GetLength(2) + _kernelSize[2];
            int totalPaddingY = (inputValues.GetLength(1) - 1) * _strides[0] - inputValues.GetLength(1) + _kernelSize[1];

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
        

        public override Array CalculateOutput(Array input)
        {
            Array inputValues = input;// as float[][][];

            if (_padding == Padding.Same)
            {
                inputValues = ZeroPad(inputValues);
            }

            int outputHeight = (inputValues.GetLength(1) - _kernelSize[1]) / _strides[0] + 1;
            int outputWidth = (inputValues.GetLength(2) - _kernelSize[2]) / _strides[1] + 1;

            float[,,] result = new float[_kernels.Count, outputHeight, outputWidth];

            for (int kernelIndex = 0; kernelIndex < _kernels.Count; kernelIndex++)
            {
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

                    for (int y = 0; y < outputHeight; y++)
                    {
                        for (int x = 0; x < outputWidth; x++)
                        {
                            result[kernelIndex, y, x] = _activationFunction.Calculate(_kernels[kernelIndex].DotProduct(inputValues, x * _strides[1], y * _strides[0]));
                        }
                    }
                }
            }

            OutputValues = result;

            return result;
        }

        public override Array BackPropagate(Array error, float learningRate)
        {
            float[,,] errorSignal = error as float[,,];

            // every output 'pixel' is the result of input * kernel
            // errorSignal has same shape as output

            for (int kernelIndex = 0; kernelIndex < errorSignal.GetLength(0); kernelIndex++)
            {
                _kernels[kernelIndex].ResetErrorSignals();

                for (int y = 0; y < errorSignal.GetLength(1); y++)
                {
                    for (int x = 0; x < errorSignal.GetLength(2); x++)
                    {
                        _kernels[kernelIndex].AddErrorSignal(errorSignal[kernelIndex, y, x], ActivationFunction.Derivative(OutputValues[kernelIndex, y, x]));
                    }
                }

                _kernels[kernelIndex].UpdateWeights(learningRate);
            }

            return error;
        }
    }
}
