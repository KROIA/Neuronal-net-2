// X-OR.cpp : Diese Datei enthält die Funktion "main". Hier beginnt und endet die Ausführung des Programms.
//

#include <iostream>
#include <vector>
#include <string>
#include "net.h"

#include "SFML/Graphics.hpp"

using std::vector;
using std::cout;
using std::string;

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

int main()
{
    Net net;

	net.setDimensions(3, 10, 100, 5);
	net.setStreamSize(1);
	net.setActivation(Activation::sigmoid);
	net.setHardware(Hardware::cpu);
	net.build();
/*	cout << net.getWeight()[0] << "\n";
	cout << net.getWeight()[1] << "\n";
	cout << net.getWeight()[2] << "\n";
	cout << net.getWeight()[3] << "\n";*/
	//getchar();
	//net.setWeight(generateWeights(net.getInputCount(), net.getHiddenXCount(), net.getHiddenYCount(), net.getOutputCount()).data());
	//setWeight(net);


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

// Programm ausführen: STRG+F5 oder Menüeintrag "Debuggen" > "Starten ohne Debuggen starten"
// Programm debuggen: F5 oder "Debuggen" > Menü "Debuggen starten"

// Tipps für den Einstieg: 
//   1. Verwenden Sie das Projektmappen-Explorer-Fenster zum Hinzufügen/Verwalten von Dateien.
//   2. Verwenden Sie das Team Explorer-Fenster zum Herstellen einer Verbindung mit der Quellcodeverwaltung.
//   3. Verwenden Sie das Ausgabefenster, um die Buildausgabe und andere Nachrichten anzuzeigen.
//   4. Verwenden Sie das Fenster "Fehlerliste", um Fehler anzuzeigen.
//   5. Wechseln Sie zu "Projekt" > "Neues Element hinzufügen", um neue Codedateien zu erstellen, bzw. zu "Projekt" > "Vorhandenes Element hinzufügen", um dem Projekt vorhandene Codedateien hinzuzufügen.
//   6. Um dieses Projekt später erneut zu öffnen, wechseln Sie zu "Datei" > "Öffnen" > "Projekt", und wählen Sie die SLN-Datei aus.


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


