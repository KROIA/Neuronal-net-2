// X-OR.cpp : Diese Datei enthält die Funktion "main". Hier beginnt und endet die Ausführung des Programms.
//

#include <iostream>
#include <vector>
#include <string>
#include <fstream>


#include "neuronalNet.h"

#include "SFML/Graphics.hpp"

using std::vector;
using std::cout;
using std::string;
using namespace NeuronalNet;
using namespace Graphics;

void printSignal(const SignalVector& sig,size_t maxSize = 10);
void printSignal(const MultiSignalVector& sig,size_t maxSize = 10);
bool signalEqual(const SignalVector& a, const SignalVector& b);
bool signalEqual(const MultiSignalVector& a, const MultiSignalVector& b);
void saveWeightsToImage(const SignalVector& w, size_t inputs, size_t hiddenX, size_t hiddenY, size_t outputs, const std::string &label);
void saveWeightsToImage(const float* list, size_t width, size_t height, const std::string& label);
void saveNetinputToImage(const float* list, size_t width, size_t height, const std::string& label);
void saveDifference(const SignalVector& a, const SignalVector& b, size_t width, size_t height, const std::string& label);

vector<float> generateWeights(size_t inputs, size_t hiddenX, size_t hiddenY, size_t outputs);
void setWeight(Net& net);

string getByteString(double bytes);

const std::string path = "export";

void xorLoop();
void printWeights(const Net* net);

int main()
{
	xorLoop();
	return 0;

    Net net;

	net.setDimensions(3, 10, 100, 5);
	net.setStreamSize(1);
	net.setActivation(Activation::sigmoid);
	net.setHardware(Hardware::cpu);
	net.build();



	net.setInputVector(SignalVector({ 1.f,1.f,1.f }));
	//net.setInputVector(MultiSignalVector({ { 0,0,0 },{ 0,1,1 },{ 0,2,0 },{ 0,2,1 },{ 1,2,0 } }));
	//net.setInputVector(MultiSignalVector{ { 1,1,1 },{ 1,1,1 },{ 1,1,1 },{ 1,1,1 },{ 1,1,1 } });
	SignalVector weightsA;
	weightsA.fill(net.getWeight(), net.getWeightSize());
	//saveWeightsToImage(weightsA, net.getInputCount(), net.getHiddenXCount(), net.getHiddenYCount(), net.getOutputCount(), "A");
	net.calculate();
	cout << "\n";
	MultiSignalVector resultA = net.getOutputStreamVector();
	MultiSignalVector netinputsA = net.getNetinputStreamVector();
	cout << "Netinput:\n";
	printSignal(netinputsA);
	cout << "Signals:\n";
	printSignal(resultA);
	

	net.setHardware(Hardware::gpu_cuda);
	net.calculate();
	
	//SignalVector weightsB;
	//weightsB.fill(net.getWeight(), net.getWeightSize());
	//saveWeightsToImage(weightsB, net.getInputCount(), net.getHiddenXCount(), net.getHiddenYCount(), net.getOutputCount(), "B");
	cout << "\n";
	MultiSignalVector resultB = net.getOutputStreamVector();
	MultiSignalVector netinputsB = net.getNetinputStreamVector();
	cout << "Netinput:\n";
	printSignal(netinputsB);
	cout << "Signals:\n";
	printSignal(resultB);
	net.setHardware(Hardware::cpu);
	net.calculate();
	cout << "\n";
	SignalVector weightsC;
	weightsC.fill(net.getWeight(), net.getWeightSize());
//	saveWeightsToImage(weightsC, net.getInputCount(), net.getHiddenXCount(), net.getHiddenYCount(), net.getOutputCount(), "C");
	MultiSignalVector resultC = net.getOutputStreamVector();
	MultiSignalVector netinputsC = net.getNetinputStreamVector();
	cout << "Netinput:\n";
	printSignal(netinputsC);
	cout << "Signals:\n";
	printSignal(resultC);


	saveNetinputToImage(netinputsA[0].begin(), net.getHiddenXCount(), net.getHiddenYCount(), "A_netinput");
	saveNetinputToImage(netinputsA[0].begin()+ net.getHiddenXCount()* net.getHiddenXCount(), 1, net.getOutputCount(), "A_netinput_out");

	saveNetinputToImage(netinputsB[0].begin(), net.getHiddenXCount(), net.getHiddenYCount(), "B_netinput");
	saveNetinputToImage(netinputsB[0].begin() + net.getHiddenXCount() * net.getHiddenXCount(), 1, net.getOutputCount(), "B_netinput_out");

	saveDifference(netinputsA[0], netinputsB[0], net.getHiddenXCount(), net.getHiddenYCount(), "Difference");


	cout << "Result:\n";
	if (signalEqual(resultA, resultB) && signalEqual(resultA, resultC))
		cout << "  result PASS\n";
	else
		cout << "  FAIL Output vectors are not equal\n";
	
	if (signalEqual(netinputsA, netinputsB) && signalEqual(netinputsA, netinputsC))
		cout << "  netinput PASS\n";
	else
		cout << "  FAIL Netinput vectors are not equal\n";

	if (signalEqual(weightsA, weightsC))
		cout << "  Weights PASS\n";
	else
		cout << "  FAIL Weights vectors are not equal\n";
}

void saveVec(const string& filename, const float *begin, size_t size)
{
	std::ofstream ofs(filename.c_str(), std::ofstream::app);

	for (size_t i = 0; i < size; ++i)
		ofs << begin[i] << ";";
	ofs << "\n";

	ofs.close();

}
void xorLoop()
{
	Display display(sf::Vector2u(1000,800),"X-OR Example");
	
	BackpropNet net;
	
	MultiSignalVector trainigsSet(4, 2);
	MultiSignalVector resultSet(4, 1);

	net.setDimensions(2, 2, 4, 1);
	net.setStreamSize(trainigsSet.size());
	net.setActivation(Activation::sigmoid);
	net.setHardware(Hardware::cpu);
	net.m_lernParameter = 0.1;
	net.build();

	sf::Vector2f spacing(40, 40);
	float neuronSize = 20;

	NetModel netModel1(&net);
	netModel1.streamIndex(0);
	netModel1.neuronSize(neuronSize);
	netModel1.pos(sf::Vector2f(100, 100));
	netModel1.neuronSpacing(spacing);
	display.addDrawable(&netModel1);

	NetModel netModel2(&net);
	netModel2.streamIndex(1);
	netModel2.neuronSize(neuronSize);
	netModel2.pos(sf::Vector2f(100, 500));
	netModel2.neuronSpacing(spacing);
	display.addDrawable(&netModel2);

	NetModel netModel3(&net);
	netModel3.streamIndex(2);
	netModel3.neuronSize(neuronSize);
	netModel3.pos(sf::Vector2f(700, 100));
	netModel3.neuronSpacing(spacing);
	display.addDrawable(&netModel3);

	NetModel netModel4(&net);
	netModel4.streamIndex(3);
	netModel4.neuronSize(neuronSize);
	netModel4.pos(sf::Vector2f(700, 500));
	netModel4.neuronSpacing(spacing);
	display.addDrawable(&netModel4);

	
	display.frameRateTarget(60);

	trainigsSet[0] = SignalVector(vector<float>{ 0,0 });
	trainigsSet[1] = SignalVector(vector<float>{ 0,1 });
	trainigsSet[2] = SignalVector(vector<float>{ 1,0 });
	trainigsSet[3] = SignalVector(vector<float>{ 1,1 });

	resultSet[0] = SignalVector(vector<float>{ 0 });
	resultSet[1] = SignalVector(vector<float>{ 1 });
	resultSet[2] = SignalVector(vector<float>{ 1 });
	resultSet[3] = SignalVector(vector<float>{ 0 });

	float averageError = 0;
	float currentError = 0;
	size_t iteration = 0;

	getchar();
	while (display.isOpen())
	{
		++iteration;
		currentError = 0;

		std::vector<float> deltaW;
		std::vector<float> deltaB;
		//for (size_t i = 0; i < trainigsSet.size(); ++i)
		{
			/*while (!display.needsFrameUpdate())
			{
				sf::sleep(sf::milliseconds(2));
				display.processEvents();
			}*/
			for (size_t i = 0; i < trainigsSet.size(); ++i)
			{
				//net.setInputVector(trainigsSet);
				net.setInputVector(i,trainigsSet[i]);
				net.calculate(i);
				//net.learn(resultSet);
				net.learn(i,resultSet[i]);
			}

			if (display.needsFrameUpdate())
			{
				display.processEvents();
				display.draw();
			}

			//SignalVector out = net.getOutputVector();
			//for(size_t i=0; i< out)

			
			/*if (iteration % 100 == 0 || iteration == 1)
			{
				if (i == 0)
				{
					deltaW = net.deltaWeight;
					deltaB = net.deltaBias;
				}
				else
				{
					for (size_t j = 0; j < deltaW.size(); ++j)
						deltaW[j] += net.deltaWeight[j];
					for (size_t j = 0; j < deltaB.size(); ++j)
						deltaB[j] += net.deltaBias[j];

					if (i == trainigsSet.size() - 1)
					{
						for (size_t j = 0; j < deltaW.size(); ++j)
							deltaW[j] /= (float)deltaW.size();
						for (size_t j = 0; j < deltaB.size(); ++j)
							deltaB[j] /= (float)deltaB.size();

						saveVec("dW.csv", deltaW.data(), deltaW.size());
						saveVec("dB.csv", deltaB.data(), deltaB.size());
						saveVec("w.csv", net.getWeight(), net.getWeightSize());
					}
				}
			}*/

			MultiSignalVector err = net.getError();
			//std::cout << "Error [" << i << "]\t";
			for (size_t j = 0; j < err.size(); ++j)
			{
				//std::cout << err[j] << "\t";
				for (size_t k = 0; k < err[j].size(); ++k)
				{
					currentError += abs(err[j][k]);
				}
			}
			
			//std::cout << "\n";

		}
		currentError /= (trainigsSet.size() * net.getOutputCount());
		//averageError = (averageError * 0.8) + (0.2 * currentError);
		averageError = currentError;
		
		if (iteration % 1000 == 0)
		{
			std::cout << "iteration [" << iteration << "]\t Error: " << averageError<<"\n";
			net.setInputVector(trainigsSet);
			net.calculate();
			//display.processEvents();
			//display.draw();
			for (size_t i = 0; i < trainigsSet.size(); ++i)
			{
				
				
				SignalVector output = net.getOutputVector(i);
				std::cout << "Set [" << i << "]\t";
				for (size_t j = 0; j < output.size(); ++j)
				{
					std::cout << output[j] << "\t";
					
				}
				std::cout << "\n";
				
			}
			getchar();
			printWeights(&net);
			
		}
	}
}

void printWeights(const Net* net)
{
	size_t iterator = 0;
	const float* weights = net->getWeight();
	std::cout << "WeightCount = " << net->getWeightSize()<<"\n";

	if (net->getHiddenYCount() * net->getHiddenXCount() == 0)
	{
		for (size_t i = 0; i < net->getOutputCount(); ++i)
		{
			std::cout << "O" << i<< " ";
			for (size_t y = 0; y < net->getInputCount(); ++y)
			{
				char str[30];
				sprintf_s(str, "%10.3f  ", weights[iterator]);
				std::cout << str;
				++iterator;
			}
			std::cout << "     ";
		}
	}
	else
	{

		for (size_t i = 0; i < net->getHiddenYCount(); ++i)
		{
			std::cout << "H0_" << i << " ";
			for (size_t y = 0; y < net->getInputCount(); ++y)
			{
				char str[30];
				sprintf_s(str, "%10.3f  ", weights[iterator]);
				std::cout << str;
				++iterator;
			}
			std::cout << "     ";
		}
		std::cout << "\n";

		for (size_t x = 1; x < net->getHiddenXCount(); ++x)
		{
			for (size_t i = 0; i < net->getHiddenYCount(); ++i)
			{
				std::cout << "H"<<x<<"_" << i << " ";
				for (size_t y = 0; y < net->getHiddenYCount(); ++y)
				{
					char str[30];
					sprintf_s(str, "%10.3f  ", weights[iterator]);
					std::cout << str;
					++iterator;
				}
				std::cout << "     ";
			}
			std::cout << "\n";
		}
		for (size_t i = 0; i < net->getOutputCount(); ++i)
		{
			std::cout << "O" << i << " ";
			for (size_t y = 0; y < net->getHiddenYCount(); ++y)
			{
				char str[30];
				sprintf_s(str, "%10.3f  ", weights[iterator]);
				std::cout << str;
				++iterator;
			}
			std::cout << "     ";
		}
	}
	std::cout << "\n";
}



void printSignal(const SignalVector& sig, size_t maxSize)
{
	std::cout << "Signals: ";
	if (sig.size() < maxSize)
		maxSize = sig.size();
	for (size_t i = 0; i < maxSize; ++i)
	{
		printf("%f\t", sig[i]);
	}
	if (maxSize < sig.size())
		printf("...");
	std::cout << "\n";
}
void printSignal(const MultiSignalVector& sig, size_t maxSize)
{
	std::cout << "Streams:\n";
	for (size_t i = 0; i < sig.size(); ++i)
	{
		printf(" Stream [%5li]\t", i);
		printSignal(sig[i], maxSize);
	}
	std::cout << "\n";
}

string getByteString(double bytes)
{
	double b = bytes;
	int exp = 0;

	while (b >= 1000.f)
	{
		b /= 1000;
		exp += 3;
	}
	switch (exp)
	{
		case 0:
			return std::to_string(b) + " bytes";
		case 3:
			return std::to_string(b) + " KB";
		case 6:
			return std::to_string(b) + " MB";
		case 9:
			return std::to_string(b) + " GB";
		case 12:
			return std::to_string(b) + " TB";
	}
	return "unknown amount";
}

bool signalEqual(const SignalVector& a, const SignalVector& b)
{
	if (a.size() != b.size())
		return false;

	for (size_t i = 0; i < a.size(); ++i)
	{
		if (abs(a[i] - b[i]) > 0.00001)
			return false;
	}
	return true;
}
bool signalEqual(const MultiSignalVector& a, const MultiSignalVector& b)
{
	if (a.size() != b.size())
		return false;

	for (size_t i = 0; i < a.size(); ++i)
	{
		if (!signalEqual(a[i], b[i]))
			return false;
	}
	return true;
}


void saveWeightsToImage(const float* list, size_t width, size_t height, const std::string& label)
{
	if (width * height == 0)
		return;
	sf::Image image;
	image.create(width, height);

	sf::Texture texture;
	texture.create(width, height);

	sf::Sprite sprite(texture); // needed to draw the texture on screen


	size_t counter = 0;
	for (size_t y = 0; y < height; ++y)
	{
		for (size_t x = 0; x < width; ++x)
		{
			//printf("%1.0f", list[y * width + x]);
			/*pixels[counter] = list[y * width + x]; // obviously, assign the values you need here to form your color
			pixels[counter + 1] = 0;
			pixels[counter + 2] = 0;
			pixels[counter + 3] = 255;*/
			float value = list[y * width + x];
			sf::Color color(0, 0, 0, 255);
			if (value > 0)
				color.g = value*255;
			else
				color.r = -value * 255;
			
			image.setPixel(x, y, color);
			//counter+=4;
		}
		//cout << "\n";
	}
	texture.loadFromImage(image);
	static int fileCounter = 0;
	cout << "Save...\n";
	image.saveToFile(path+"\\"+label + "__mat_" + std::to_string(width) + "_" + std::to_string(fileCounter) + ".jpg");
	fileCounter++;
}
void saveNetinputToImage(const float* list, size_t width, size_t height, const std::string& label)
{
	if (width * height == 0)
		return;
	sf::Image image;
	image.create(width, height);

	sf::Texture texture;
	texture.create(width, height);

	sf::Sprite sprite(texture); // needed to draw the texture on screen


	size_t counter = 0;
	for (size_t y = 0; y < height; ++y)
	{
		for (size_t x = 0; x < width; ++x)
		{
			//printf("%1.0f", list[y * width + x]);
			/*pixels[counter] = list[y * width + x]; // obviously, assign the values you need here to form your color
			pixels[counter + 1] = 0;
			pixels[counter + 2] = 0;
			pixels[counter + 3] = 255;*/
			float value = list[x * height + y];
			sf::Color color(0, 0, 0, 255);
			if (value > 0)
				color.g = value * 255;
			else
				color.r = -value * 255;

			image.setPixel(x, y, color);
			//counter+=4;
		}
		//cout << "\n";
	}
	texture.loadFromImage(image);
	static int fileCounter = 0;
	cout << "Save...\n";
	image.saveToFile(path + "\\" + label + "__mat_" + std::to_string(width) + "_" + std::to_string(fileCounter) + ".jpg");
	fileCounter++;
}
void saveWeightsToImage(const SignalVector& w, size_t inputs, size_t hiddenX, size_t hiddenY, size_t outputs, const std::string& label)
{
	size_t layers = hiddenX + 1;
	if (hiddenY == 0)
		layers = 1;

	float* weights = w.begin();

	if (hiddenX * hiddenY == 0)
	{
		saveWeightsToImage(weights, inputs, outputs, label+"__input_To_Output_Layer");
	}
	else
	{
		saveWeightsToImage(weights, inputs, hiddenY, label+"__input_To_Hidden_Layer");
		weights += inputs * hiddenY;
		for (size_t i = 1; i < hiddenX; ++i)
		{
			saveWeightsToImage(weights, hiddenY, hiddenY, label+"__Hidden_"+std::to_string(i)+"_To_Hidden_Layer");
			weights += hiddenY * hiddenY;
		}
		saveWeightsToImage(weights, hiddenY, outputs, label+"__Hidden_" + std::to_string(hiddenX) + "_To_Output_Layer");
	}
}

void saveDifference(const SignalVector& a, const SignalVector& b, size_t width, size_t height, const std::string& label)
{
	SignalVector res(a.size());
	for (size_t i = 0; i < a.size(); ++i)
	{
		res[i] = sqrt(pow(a[i] - b[i],2))*100.f;
		//res[i] = a[i] - b[i];
		//if (res[i] < 0)
		//	res[i] = -res[i];
		
		//if(res[i]>0.1)
		//	cout << res[i]<<"\n";
	}
	saveNetinputToImage(res.begin(), width, height, label);
	
	SignalVector res2 = res;
	for (size_t i = 0; i < a.size(); ++i)
	{
		res2[i] /= 100;
	}
	saveNetinputToImage(res2.begin(), width, height, label+"_scale0.01");

	for (size_t i = 0; i < a.size(); ++i)
	{
		res[i] *= 100;
	}
	saveNetinputToImage(res.begin(), width, height, label+"_scale10");
	for (size_t i = 0; i < a.size(); ++i)
	{
		res[i] *= 10;
	}
	saveNetinputToImage(res.begin(), width, height, label + "_scale100");
}

vector<float> generateWeights(size_t inputs, size_t hiddenX, size_t hiddenY, size_t outputs)
{
	size_t weightSize = inputs * hiddenY + hiddenX * hiddenY * hiddenY + hiddenY * outputs;
	vector<float> weights(weightSize, 0);

	size_t index = 0;

	for (size_t x = 0; x < inputs; ++x)
	{
		for (size_t y = 0; y < hiddenY; ++y)
		{

			weights[index] = 1;
			++index;
		}
	}

	
	for (size_t l = 1; l < hiddenX; ++l)
	{
		
		for (size_t x = 0; x < hiddenY; ++x)
		{
			float count = 0.0;
			for (size_t y = 0; y < hiddenY; ++y)
			{
				//weights[y * hiddenX + x] = (sin((float)x*3.f/ (float)hiddenX) + sin((float)x * 3.f / (float)hiddenY))/2.f;
				//weights[index] = count;
				//count += 0.1;
				weights[index] = sin(((float)x * 3.f) / (float)hiddenX);
				//weights[index] = 1;
				++index;
			}
		}
	}
	for (size_t i = 0; i < hiddenY * outputs; ++i)
	{
		weights[index] = 1;
		++index;
	}
	return weights;
}

void setWeight(Net& net)
{
	size_t inputs = net.getInputCount();
	size_t hiddenX = net.getHiddenXCount();
	size_t hiddenY = net.getHiddenYCount();
	size_t outputs = net.getOutputCount();

	for (size_t i = 0; i < hiddenY; ++i)
	{
		for (size_t inp = 0; inp < inputs; ++inp)
			net.setWeight(0, i, inp, 1);
	}

	for (size_t x = 1; x < hiddenX; ++x)
	{
		for (size_t y = 0; y < hiddenY; ++y)
		{
			if(y%2==0)
				for (size_t inp = 0; inp < hiddenY; ++inp)
					net.setWeight(x, y, inp,1- (float)(rand() % 100) / 100.f);
			else
				for (size_t inp = 0; inp < hiddenY; ++inp)
					net.setWeight(x, y, inp,-1+ (float)(rand() % 100) / 100.f);
		}
	}

	for (size_t y = 0; y < outputs; ++y)
	{
		for (size_t inp = 0; inp < hiddenY; ++inp)
			net.setWeight(hiddenX, y, inp, 1);
	}
}


