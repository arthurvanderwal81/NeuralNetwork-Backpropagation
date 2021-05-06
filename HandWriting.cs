using System;
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;
using System.Collections.Generic;
using System.Runtime.InteropServices;

using NeuralNetLib;
using NeuralNetLib.Layers;
using NeuralNetLib.Helpers;
using NeuralNetLib.ActivationFunctions;

namespace NeuralNetwork___Backpropagation
{
    public class HandWriting
    {
        public class HandWrittenImage
        {
            public int Width
            {
                get
                {
                    return ImageData.GetLength(1);
                }
            }

            public int Height
            {
                get
                {
                    return ImageData.GetLength(0);
                }
            }

            public byte[,] ImageData { get; }
            public float[,,] NormalizedImageData { get; private set; }
            public float[] OneHotVector { get; private set; }

            public byte Label { get; }

            private void GenerateNormalizedImageData()
            {
                // Hardcoded 1 channel
                NormalizedImageData = new float[1, ImageData.GetLength(0), ImageData.GetLength(1)];

                for (int y = 0; y < ImageData.GetLength(1); y++)
                {
                    for (int x = 0; x < ImageData.GetLength(0); x++)
                    {
                        NormalizedImageData[0, y, x] = (float)ImageData[y, x] / 255.0f;
                    }
                }
            }

            private void GenerateOneHotVector()
            {
                // Hardcoded 10 classes
                OneHotVector = new float[10];
                OneHotVector[Label] = 1.0f;
            }

            public HandWrittenImage(byte[,] imageData, byte label)
            {
                ImageData = imageData;
                Label = label;

                GenerateNormalizedImageData();
                GenerateOneHotVector();
            }
        }

        private const string _trainingImageDataPath = @"E:\ML\MNIST\train\train-images.idx3-ubyte";
        private const string _trainingLabelDataPath = @"E:\ML\MNIST\train\train-labels.idx1-ubyte";

        private static Bitmap CopyDataToBitmap(byte[] data)
        {
            //Here create the Bitmap to the know height, width and format
            Bitmap bmp = new Bitmap(28, 28, PixelFormat.Format8bppIndexed);

            //Create a BitmapData and Lock all pixels to be written 
            BitmapData bmpData = bmp.LockBits(
                                 new Rectangle(0, 0, bmp.Width, bmp.Height),
                                 ImageLockMode.WriteOnly, bmp.PixelFormat);

            //Copy the data from the byte array into BitmapData.Scan0
            Marshal.Copy(data, 0, bmpData.Scan0, data.Length);


            //Unlock the pixels
            bmp.UnlockBits(bmpData);


            //Return the bitmap 
            return bmp;
        }

        private static int GetHighEndianInt32(byte[] bytes)
        {
            return bytes[0] << 32 | bytes[1] << 16 | bytes[2] << 8 | bytes[3];
        }

        // http://yann.lecun.com/exdb/mnist/
        private static List<HandWrittenImage> ReadImages(string path, byte[] labels)
        {
            FileStream fileStream = File.Open(path, FileMode.Open);

            BinaryReader binaryReader = new BinaryReader(fileStream);

            int magicNumber = GetHighEndianInt32(binaryReader.ReadBytes(4));

            Console.WriteLine(string.Format("0x{0:X8}", magicNumber));

            if (magicNumber == 0x00000803)
            {
                Console.WriteLine("Verified Magic Number");
            }
            else
            {
                Console.WriteLine(string.Format("Magic Number mismatch, found: 0x{0:X8}", magicNumber));
            }

            int numberOfImages = GetHighEndianInt32(binaryReader.ReadBytes(4));
            int numberofRows = GetHighEndianInt32(binaryReader.ReadBytes(4));
            int numberOfColumns = GetHighEndianInt32(binaryReader.ReadBytes(4));

            Console.WriteLine("Number of Images: {0}", numberOfImages);
            Console.WriteLine(string.Format("Image dimensions: {0}x{1}", numberOfColumns, numberofRows));

            List<HandWrittenImage> result = new List<HandWrittenImage>();

            for (int i = 0; i < numberOfImages; i++)
            {
                byte[,] imageData = new byte[numberofRows, numberOfColumns];
                byte[] imageDataL = new byte[numberofRows * numberOfColumns];

                for (int row = 0; row < numberofRows; row++)
                {
                    for (int column = 0; column < numberOfColumns; column++)
                    {
                        imageData[row, column] = binaryReader.ReadByte();
                        imageDataL[row * numberOfColumns + column] = imageData[row, column];
                    }
                }

                HandWrittenImage image = new HandWrittenImage(imageData, labels[i]);
                result.Add(image);

                //Bitmap bmp = CopyDataToBitmap(imageDataL);
                //bmp.Save(string.Format(@"E:\IMG{0} - {1}.BMP", i, labels[i]));
            }

            return result;
        }

        private static byte[] ReadLabels(string path)
        {
            FileStream fileStream = File.Open(path, FileMode.Open);

            BinaryReader binaryReader = new BinaryReader(fileStream);

            //byte[] magicNumberBytes = binaryReader.ReadBytes(4);
            int magicNumber = GetHighEndianInt32(binaryReader.ReadBytes(4)); //magicNumberBytes[0] << 32 | magicNumberBytes[1] << 16 | magicNumberBytes[2] << 8 | magicNumberBytes[3];

            Console.WriteLine(string.Format("0x{0:X8}", magicNumber));

            if (magicNumber == 0x00000801)
            {
                Console.WriteLine("Verified Magic Number");
            }
            else
            {
                Console.WriteLine(string.Format("Magic Number mismatch, found: 0x{0:X8}", magicNumber));
            }

            int numberOfImages = GetHighEndianInt32(binaryReader.ReadBytes(4));

            List<byte> result = new List<byte>();

            for (int i = 0; i < numberOfImages; i++)
            {
                result.Add(binaryReader.ReadByte());
            }

            return result.ToArray();
        }

        public static void mainTraining()
        {
            Model model = new Model();

            model.Add(new ConvolutionalLayer2D(1, 32, new int[] { 1, 5, 5 }, new int[] { 1, 1 }, ConvolutionalLayer2D.Padding.Same, new ReLuActivationFunction()));
            model.Add(new MaxPoolingLayer(2));
            model.Add(new ConvolutionalLayer2D(3, 64, new int[] { 32, 5, 5 }, new int[] { 1, 1 }, ConvolutionalLayer2D.Padding.Same, new ReLuActivationFunction()));
            model.Add(new MaxPoolingLayer(4));
            model.Add(new FlattenLayer(5));
            model.Add(new FullyConnectedLayer(6, 128, new ReLuActivationFunction()));
            model.Add(new DropoutLayer(7, 0.4f));
            model.Add(new FullyConnectedLayer(8, 10, new SigmoidActivationFunction()));

            byte[] labels = ReadLabels(_trainingLabelDataPath);
            List<HandWrittenImage> images = ReadImages(_trainingImageDataPath, labels);

            Model.TrainingData testImages = new Model.TrainingData();

            foreach (HandWrittenImage image in images)
            {
                testImages.InputData.Add(image.NormalizedImageData);
                testImages.ExpectedOutputData.Add(image.OneHotVector);

                break;// 1 image
            }
            
            // 2000 epochs on single image to test convergence
            model.Train(testImages, 2000, 1);
        }

        public static void main()
        {
            Model.TrainingData training = new Model.TrainingData();

            /*
            training.InputData.Add(new float[,,]
            {
                {
                    { 1.0f / 10.0f, 2.0f / 10.0f, 3.0f / 10.0f},
                    { 4.0f / 10.0f, 5.0f / 10.0f, 6.0f / 10.0f},
                    { 7.0f / 10.0f, 8.0f / 10.0f, 9.0f / 10.0f },
                }
            });
            */

            training.InputData.Add(RandomHelper.GetRandom3DArray(new int[] { 1, 6, 6 }));

            training.ExpectedOutputData.Add(new float[] { 1.0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f });

            Model model = new Model();

            model.Add(new ConvolutionalLayer2D(1, 1, new int[] { 1, 3, 3 }, new int[] { 1, 1 }, ConvolutionalLayer2D.Padding.None, new SigmoidActivationFunction()));
            model.Add(new FlattenLayer(2));

            model.Train(training, 500000, 1);
        }
    }
}
