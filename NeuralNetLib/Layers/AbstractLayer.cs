using System;

namespace NeuralNetLib.Layers
{
    public abstract class AbstractLayer
    {
        public abstract Array CalculateOutput(Array input);

        public abstract Array BackPropagate(Array error, float learningRate);
    }
}
