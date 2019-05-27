using System;
using System.IO;
using System.Drawing;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

using NeuralNetLib.Layers;

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
            public float[,] NormalizedImageData { get; private set; }

            public byte Label { get; }

            private void GenerateNormalizedImageData()
            {
                NormalizedImageData = new float[ImageData.GetLength(0), ImageData.GetLength(1)];

                for (int y = 0; y < ImageData.GetLength(1); y++)
                {
                    for (int x = 0; x < ImageData.GetLength(0); x++)
                    {
                        NormalizedImageData[y, x] = (float)ImageData[y, x] / 255.0f;
                    }
                }
            }

            public HandWrittenImage(byte[,] imageData, byte label)
            {
                ImageData = imageData;
                Label = label;

                GenerateNormalizedImageData();
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

        public static void main()
        {
            //byte[] labels = ReadLabels(_trainingLabelDataPath);
            //List<HandWrittenImage> images = ReadImages(_trainingImageDataPath, labels);

            //https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53

            ConvolutionalLayer2D convolutionalLayer2D = new ConvolutionalLayer2D(0, new int[] { 3, 6, 6 }, 1, new int[] { 3, 3, 3 }, new int[] { 1, 1 }, ConvolutionalLayer2D.Padding.None, null);

            convolutionalLayer2D.OutputValues = new List<float[,]>();

            convolutionalLayer2D.OutputValues.Add(new float[,]
            {
                { 12.0f, 20.0f, 30.0f, 0.0f },
                { 8.0f, 12.0f, 2.0f, 0.0f },
                { 34.0f, 70.0f, 37.0f, 4.0f },
                { 112.0f, 100.0f, 25.0f, 12.0f },
            });

            MaxPoolingLayer maxPoolingLayer = new MaxPoolingLayer(1, convolutionalLayer2D);
            maxPoolingLayer.CalculateOutput();

            /*

            convolutionalLayer2D.OutputValues.Add(new float[,]
            {
                { 0, 1, 2, 3 },
                { 4, 5, 6, 7 }
            });

            convolutionalLayer2D.OutputValues.Add(new float[,]
            {
                { 8, 9, 10, 11 },
                { 12, 13, 14, 15 }
            });

            convolutionalLayer2D.OutputValues.Add(new float[,]
            {
                { 16, 17, 18, 19 },
                { 20, 21, 22, 23 }
            });

            FlattenLayer flattenLayer = new FlattenLayer(1, convolutionalLayer2D);
            flattenLayer.CalculateOutput();

            */

            /*
            for (int channel = 0; channel < layer.InputValues.GetLength(0); channel++)
            {
                for (int y = 0; y < layer.InputValues.GetLength(1); y++)
                {
                    for (int x = 0; x < layer.InputValues.GetLength(2); x++)
                    {
                        layer.InputValues[channel, y, x] = channel + 1.0f;
                    }
                }
            }

            layer.InputValues = new float[,,]
            {
                {
                    {
                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
                    },
                    {
                        0.0f, 156.0f, 155.0f, 156.0f, 158.0f, 158.0f
                    },
                    {
                        0.0f, 153.0f, 154.0f, 157.0f, 159.0f, 159.0f
                    },
                    {
                        0.0f, 149.0f, 151.0f, 155.0f, 158.0f, 159.0f
                    },
                    {
                        0.0f, 146.0f, 146.0f, 149.0f, 153.0f, 158.0f
                    },
                    {
                        0.0f, 145.0f, 143.0f, 143.0f, 148.0f, 158.0f
                    }
                },
                {
                    {
                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
                    },
                    {
                        0.0f, 167.0f, 166.0f, 167.0f, 169.0f, 169.0f
                    },
                    {
                        0.0f, 164.0f, 165.0f, 168.0f, 170.0f, 170.0f
                    },
                    {
                        0.0f, 160.0f, 162.0f, 166.0f, 169.0f, 170.0f
                    },
                    {
                        0.0f, 156.0f, 156.0f, 159.0f, 163.0f, 168.0f
                    },
                    {
                        0.0f, 155.0f, 153.0f, 153.0f, 158.0f, 168.0f
                    }
                },
                {
                    {
                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
                    },
                    {
                        0.0f, 163.0f, 162.0f, 163.0f, 165.0f, 165.0f
                    },
                    {
                        0.0f, 160.0f, 161.0f, 164.0f, 166.0f, 166.0f
                    },
                    {
                        0.0f, 156.0f, 158.0f, 162.0f, 165.0f, 166.0f
                    },
                    {
                        0.0f, 155.0f, 155.0f, 158.0f, 162.0f, 167.0f
                    },
                    {
                        0.0f, 154.0f, 152.0f, 152.0f, 157.0f, 167.0f
                    }
                }
            };

            layer._kernels[0].Weights = new float[,,]
            {
                {
                    { -1.0f, -1.0f, 1.0f },
                    { 0.0f, 1.0f, -1.0f },
                    { 0.0f, 1.0f, 1.0f }
                },
                {
                    { 1.0f, 0.0f, 0.0f },
                    { 1.0f, -1.0f, -1.0f },
                    { 1.0f, 0.0f, -1.0f }
                },
                {
                    { 0.0f, 1.0f, 1.0f },
                    { 0.0f, 1.0f, 0.0f },
                    { 1.0f, -1.0f, 1.0f }
                }
            };

            layer._kernels[0].Bias = 1.0f;

            layer.CalculateOutput();*/
        }
    }
}
