

#include <iostream>
#include <vector>
#include <string>
#include "SFML/Graphics.hpp"

#include "neuronalNet.h"
#include "dataset.h"


using std::vector;
using std::cout;
using std::string;
using namespace NeuronalNet;
using namespace Graphics;


size_t trainingIteration = 0;
float trainingError = 1;

void train(BackpropNet& net, Dataset& dataset, size_t iterations);
void verify(BackpropNet* net, Dataset* dataset ,size_t beginDataset, size_t endDataset);

int main()
{
    int scaleDivisor = 2;
    Dataset dataset(scaleDivisor);
    const size_t xImageRaw = 28;
    const size_t yImageRaw = 28;
    const size_t xImage = xImageRaw /scaleDivisor;
    const size_t yImage = yImageRaw /scaleDivisor;
    const size_t imageSize = xImage * yImage;
    if (!dataset.importBinaryData("Dataset\\data0", xImageRaw, yImageRaw, Label::zero) ||
        !dataset.importBinaryData("Dataset\\data1", xImageRaw, yImageRaw, Label::one) ||
        !dataset.importBinaryData("Dataset\\data2", xImageRaw, yImageRaw, Label::two) ||
        !dataset.importBinaryData("Dataset\\data3", xImageRaw, yImageRaw, Label::three) ||
        !dataset.importBinaryData("Dataset\\data4", xImageRaw, yImageRaw, Label::four) ||
        !dataset.importBinaryData("Dataset\\data5", xImageRaw, yImageRaw, Label::five) ||
        !dataset.importBinaryData("Dataset\\data6", xImageRaw, yImageRaw, Label::six) ||
        !dataset.importBinaryData("Dataset\\data7", xImageRaw, yImageRaw, Label::seven) ||
        !dataset.importBinaryData("Dataset\\data8", xImageRaw, yImageRaw, Label::eight) ||
        !dataset.importBinaryData("Dataset\\data9", xImageRaw, yImageRaw, Label::nine))
    {
        cout << "Can't load Dataset";
        return -1;
    }
    dataset.shuffle();


    BackpropNet net;
    net.setDimensions(imageSize, 1, 20, Label::count);
    net.setLearnParameter(0.1);
    size_t trainingsEndIndex = 200; // dataset.count() - dataset.count() / 10;
   // net.setStreamSize(trainingsEndIndex);
    net.setActivation(Activation::sigmoid);
    net.setHardware(Hardware::cpu);
    net.build();
    
    // Training:


    MultiSignalVector inputs(trainingsEndIndex, imageSize);
    MultiSignalVector expected(trainingsEndIndex, Label::count);
    for (size_t i = 0; i < trainingsEndIndex; ++i)
    {
        inputs[i] = dataset.getImage(i).getPixels();
        expected[i] = dataset.getImage(i).getLabelVector();
    }
    cout << "setInputs\n";
    //net.setExpectedOutput(expected);
    //net.setInputVector(inputs);

    //Display display(sf::Vector2u(1000, 800), "window");
    sf::Image image;
    sf::Texture texture;
    image.create(xImage, yImage);
    texture.create(xImage, yImage);
    // Create a sprite that will display the texture
    sf::Sprite sprite(texture);
    
    for (size_t i = 0; i < 20; ++i)
    {
        const ImageGray& im = dataset.getImage(i);
        for (size_t x = 0; x < xImage; ++x)
        {
            for (size_t y = 0; y < yImage; ++y)
            {
                uint8_t col = (im.getPixel(x, y) + 1.f) * 128.f;
                image.setPixel(x, y, sf::Color(col, col, col));
            }
        }

        texture.update(image);
        image.saveToFile("Im_" + std::to_string(i) + "_" + std::to_string(im.getLabel()) +".png");
        
    }

    
    while (trainingError > 0.001)
    {
        size_t loopSize = 100;
        trainingError = 0;
        for (size_t i = 0; i < loopSize; ++i)
        {
           // cout << "calculate\t";
            
            net.setInputVector(dataset.getImage(i).getPixels());
            net.calculate();
            //cout << "learn\t";
            net.learn(dataset.getImage(i).getLabelVector());
            trainingError += net.getError().getRootMeanSquare();
            
            
        }
        ++trainingIteration;
        trainingError /= (float)loopSize;
        cout << "Error: " << trainingError << "\n";
       // train(net, dataset, 100);

        // Check training:
        if(trainingIteration%1000 == 0)
        verify(&net, &dataset, dataset.count() - 20, dataset.count());
    }
    

    

    cout << "\nExit";
    return 0;
}

void train(BackpropNet& net, Dataset& dataset, size_t iterations)
{
    for(size_t i=0; i< iterations; ++i)
    {
        cout << "calculate\t";
        net.calculate();
        cout << "learn\t";
        net.learn();
        trainingError = net.getError().getRootMeanSquare();
        cout << "Error: " << trainingError << "\n";
        ++trainingIteration;
    } 
}
void verify(BackpropNet* net, Dataset* dataset, size_t beginDataset, size_t endDataset)
{
    for (size_t i = beginDataset; i < endDataset; ++i)
    {
        net->setInputVector(dataset->getImage(i).getPixels());
        net->calculate();
        SignalVector out = net->getOutputVector();
        Label highestPrediction = Label::count;
        float highestValue = -1;
        for (size_t j = 0; j < out.size(); ++j)
        {
            if (out[j] > highestValue)
            {
                highestPrediction = (Label)j;
                highestValue = out[j];
            }
        }
        cout << "Dataset Index [" << i << "]\tLabel: " <<
            Dataset::labelToString(dataset->getImage(i).getLabel()) <<
            "\t Predicted: " << Dataset::labelToString(highestPrediction) << "\tOutputs: {";
        for (size_t j = 0; j < out.size(); ++j)
        {
            cout << out[j];
            if (j < out.size() - 1)
                cout << ", ";
        }
        cout << "}\n";
    }
    net->setInputVector(dataset->getImage(0).getPixels());
}