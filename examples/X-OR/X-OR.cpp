// X-OR.cpp : Diese Datei enthält die Funktion "main". Hier beginnt und endet die Ausführung des Programms.
//

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <utility>
#include <thread>
#include <chrono>
#include <mutex>


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

struct BenchmarkData
{
	size_t net_xSize;
	size_t net_ySize;

	bool displayEnable;
	bool enableDebugOutput;

	double minLearnTime;
	double maxLearnTime;
	double averageLearnTime;
};
std::mutex mutex;
int activeThreads = 0;
void xorBenchmarkMain();
void xorBenchmark(size_t maxIteration, BenchmarkData& data);
void xorBenchmarkThreaded(size_t thredID, BenchmarkData* list, size_t size);
void printWeights(Net* net);

int main()
{
	//vector<float> test1({ 0.1,0.2,-10.3,0.4,-0.1,-0.2,0.9 });

	//std::cout << "min: " << test1[getMinIndex<float>(test1.data(), test1.size())];
	//std::cout << "  max: " << test1[getMaxIndex<float>(test1.data(), test1.size())];

	//getchar();
	
	//xorBenchmarkMain();
	xorLoop();

	/*
	{
		using namespace std::chrono_literals;
		Debug::DebugFuncStackTimeTrace trace("init");
		
		//std::this_thread::sleep_for(100ms);
		
		{
			Debug::DebugFuncStackTimeTrace trace1("t1");
			//std::this_thread::sleep_for(100ms);
			{
				Debug::DebugFuncStackTimeTrace trace1("t2");
				//std::this_thread::sleep_for(100ms);
				{
					Debug::DebugFuncStackTimeTrace trace1("t3");
					//std::this_thread::sleep_for(100ms);
					{
						Debug::DebugFuncStackTimeTrace trace1("t4");
						//std::this_thread::sleep_for(10ms);

					}
				}
				{
					Debug::DebugFuncStackTimeTrace trace1("t5");
					std::this_thread::sleep_for(100ms);

				}
				{
					Debug::DebugFuncStackTimeTrace trace1("t6");
					std::this_thread::sleep_for(100ms);
					{
						Debug::DebugFuncStackTimeTrace trace1("t7");
						std::this_thread::sleep_for(100ms);

					}
				}
			}
		}
	}*/
	return 0;
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
	Display display(sf::Vector2u(1900,900),"X-OR Example");

	
	BackpropNet net;
	
	MultiSignalVector trainigsSet(4, 2);
	MultiSignalVector resultSet(4, 1);

	net.setDimensions(2, 2, 5, 1);
	net.setStreamSize(trainigsSet.size());
	net.setActivation(Activation::sigmoid);
	net.setHardware(Hardware::cpu);
	net.setLearnParameter(1.0);
	net.enableBias(true);
	net.build();

	sf::Vector2f spacing(80, 20);
	float neuronSize = 15;


	size_t visualConfig = NetModel::getStandardVisualConfiguration();
	NetModel netModel1(&net);
	netModel1.setStreamIndex(0);
	netModel1.setNeuronSize(neuronSize);
	netModel1.setPos(sf::Vector2f(100, 100));
	netModel1.setNeuronSpacing(spacing);
	netModel1.setVisualConfiguration(visualConfig);
	display.addDrawable(&netModel1);

	NetModel netModel2(&net);
	netModel2.setStreamIndex(1);
	netModel2.setNeuronSize(neuronSize);
	netModel2.setPos(sf::Vector2f(100, 500));
	netModel2.setNeuronSpacing(spacing);
	netModel2.setVisualConfiguration(visualConfig);
	display.addDrawable(&netModel2);

	NetModel netModel3(&net);
	netModel3.setStreamIndex(2);
	netModel3.setNeuronSize(neuronSize);
	netModel3.setPos(sf::Vector2f(1100, 100));
	netModel3.setNeuronSpacing(spacing);
	netModel3.setVisualConfiguration(visualConfig);
	display.addDrawable(&netModel3);

	NetModel netModel4(&net);
	netModel4.setStreamIndex(3);
	netModel4.setNeuronSize(neuronSize);
	netModel4.setPos(sf::Vector2f(1100, 500));
	netModel4.setNeuronSpacing(spacing);
	netModel4.setVisualConfiguration(visualConfig);
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

	float currentError = 0;
	size_t iteration = 0;

	Debug::Timer trainigTimer;
	Debug::Timer learnTimer;
	trainigTimer.start();
	MultiSignalVector err;
	while (display.isOpen())
	{
		bool frameUpdate = display.needsFrameUpdate();
		
		if (frameUpdate)
		{
			++iteration;

			trainigTimer.unpause();
			net.setInputVector(trainigsSet);
			net.calculate();
			if(iteration == 1)
				learnTimer.start();
			learnTimer.unpause();
			net.learn(resultSet);
			learnTimer.pause();
			trainigTimer.pause();

			err = net.getError();
			currentError = err.getRootMeanSquare();
		}
		net.setLearnParameter(currentError);

		if (frameUpdate)
		{
			
			/*Hardware h = net.getHardware();
			if (h != Hardware::cpu)
			{
				net.setHardware(Hardware::cpu);
			}*/
			display.processEvents();
			display.draw();
			//net.setHardware(h);
			
		}

		

		static Debug::Timer timer(true);
		//if (timer.getMillis() > 1000)
		if (iteration%100 == 0 && frameUpdate)
		{
			std::cout << "iteration [" << iteration << "]\t Error: " << currentError << "\tTrainigtime: " 
				      << trainigTimer.getMillis() << " ms\tlearnTime: "<< learnTimer.getMicros()/(double)iteration<< " us/training " <<"\n";
			timer.reset();
			timer.start();
		}
		/*Hardware h = net.getHardware();
		if (h != Hardware::cpu)
		{
			net.setHardware(Hardware::cpu);
		}
		printWeights(&net);
		net.setHardware(h);
		getchar();*/
		if (currentError < 0.05 && frameUpdate)
		{
			std::cout << "iteration [" << iteration << "]\t Error: " << currentError << "\tTrainigtime: "
				      << trainigTimer.getMillis() << "ms learnTime: " << learnTimer.getMicros() / (double)iteration << "us/training\n";
			//net.setHardware(Hardware::gpu_cuda);
			net.setInputVector(trainigsSet);
			net.calculate();
			for (size_t i = 0; i < trainigsSet.size(); ++i)
			{
				SignalVector output = net.getOutputVector(i);
				std::cout << "CPU Set [" << i << "]\t";
				for (size_t j = 0; j < output.size(); ++j)
				{
					std::cout << output[j] << "\t";
					
				}
				std::cout << "\n";
			}
			std::cout << "\n";
			/*net.setHardware(Hardware::gpu_cuda);
			net.setInputVector(trainigsSet);
			net.calculate();
			for (size_t i = 0; i < trainigsSet.size(); ++i)
			{
				SignalVector output = net.getOutputVector(i);
				std::cout << "GPU Set [" << i << "]\t";
				for (size_t j = 0; j < output.size(); ++j)
				{
					std::cout << output[j] << "\t";

				}
				std::cout << "\n";
			}
			net.setHardware(Hardware::cpu);
			printWeights(&net);*/
			getchar();
			
		}
	}
}

void xorBenchmarkMain()
{
	size_t minX = 1;
	size_t minY = 1;
	size_t maxX = 20;
	size_t maxY = 20;

	size_t arraySizeX = maxX - minX;
	size_t arraySizeY = maxY - minY;
	size_t arraySize = arraySizeX * arraySizeY;

	BenchmarkData* benchDataList = new BenchmarkData[arraySize];


	for (size_t x = 0; x < arraySizeX; ++x)
	{
		for (size_t y = 0; y < arraySizeY; ++y)
		{
			benchDataList[x * arraySizeY + y].net_xSize = x + minX;
			benchDataList[x * arraySizeY + y].net_ySize = y + minY;
			benchDataList[x * arraySizeY + y].displayEnable = false;
			benchDataList[x * arraySizeY + y].enableDebugOutput = false;
		}
	}

	size_t jobSize = 1;
	size_t nextJobStart = 0;

	bool jobDone = false;
	using namespace std::chrono_literals;
	std::cout << "Start\n";
	while (!jobDone)
	{
		int currentRunningThreads = 0;
		{
			std::lock_guard<std::mutex> lockGuard(mutex);
			currentRunningThreads = activeThreads;
		}
		if (nextJobStart >= arraySize)
			jobDone = true;
		if (currentRunningThreads < std::thread::hardware_concurrency())
		{
			if (nextJobStart + jobSize > arraySize)
				jobSize = arraySize - nextJobStart;
			std::cout << "Starte Job: " << nextJobStart << " - " << nextJobStart + jobSize << "  " << (float)(nextJobStart + jobSize) * 100.f / (float)arraySize << "\n";
			std::thread thread(xorBenchmarkThreaded, 0, &benchDataList[nextJobStart], jobSize);
			thread.detach();
			nextJobStart += jobSize;
		}
		std::this_thread::sleep_for(10ms);
		
	}
	int currentRunningThreads = 0;
	int currentRunningThreads1 = 0;
	Debug::Timer timer;
	timer.start();
	do {
		{
			std::lock_guard<std::mutex> lockGuard(mutex);
			currentRunningThreads = activeThreads;
		}
		std::this_thread::sleep_for(100ms);
		if (currentRunningThreads1 != currentRunningThreads || timer.getMillis() > 1000)
		{
			std::cout << "Current running threads: " << currentRunningThreads << "\n";
			currentRunningThreads1 = currentRunningThreads;
			timer.reset();
			timer.start();
		}
	} while (currentRunningThreads > 0);
	std::cout << "Jobs done\n";

	{
		std::ofstream myfile;

		std::cout << "Open file\n";
		myfile.open("averageLearnTime.csv");

		std::cout << "Write file\n";
		myfile << ";";


		for (size_t x = minX; x < maxX; ++x)
		{
			myfile << x << ";";
		}
		myfile << "\n";

		for (size_t y = 0; y < arraySizeY; ++y)
		{
			myfile << y + minY << ";";
			std::cout << "store line: " << y << "\n";
			for (size_t x = 0; x < arraySizeX; ++x)
			{
				myfile << benchDataList[x * arraySizeX + y].averageLearnTime << ";";
			}
			myfile << "\n";
		}
		myfile << "\n";
		std::cout << "closeFile\n";
		myfile.close();
	}
	{
		std::ofstream myfile;

		std::cout << "Open file\n";
		myfile.open("minLearnTime.csv");

		std::cout << "Write file\n";
		myfile << ";";


		for (size_t x = minX; x < maxX; ++x)
		{
			myfile << x << ";";
		}
		myfile << "\n";

		for (size_t y = 0; y < arraySizeY; ++y)
		{
			myfile << y + minY << ";";
			std::cout << "store line: " << y << "\n";
			for (size_t x = 0; x < arraySizeX; ++x)
			{
				myfile << benchDataList[x * arraySizeX + y].minLearnTime << ";";
			}
			myfile << "\n";
		}
		myfile << "\n";
		std::cout << "closeFile\n";
		myfile.close();
	}
	{
		std::ofstream myfile;

		std::cout << "Open file\n";
		myfile.open("maxLearnTime.csv");

		std::cout << "Write file\n";
		myfile << ";";


		for (size_t x = minX; x < maxX; ++x)
		{
			myfile << x << ";";
		}
		myfile << "\n";

		for (size_t y = 0; y < arraySizeY; ++y)
		{
			myfile << y + minY << ";";
			std::cout << "store line: " << y << "\n";
			for (size_t x = 0; x < arraySizeX; ++x)
			{
				myfile << benchDataList[x * arraySizeX + y].maxLearnTime << ";";
			}
			myfile << "\n";
		}
		myfile << "\n";
		std::cout << "closeFile\n";
		myfile.close();
	}
	std::cout << "exit\n";
}
void xorBenchmarkThreaded(size_t thredID, BenchmarkData* list, size_t size)
{
	
	{
		std::lock_guard<std::mutex> lockGuard(mutex);
		++activeThreads;
	}
	for (size_t y = 0; y < size; ++y)
	{
		//std::cout << "Thread: " << thredID << " " << (float)y*100.f / (float)size << "%\n";
		xorBenchmark(10, list[y]);
	}
	{
		std::lock_guard<std::mutex> lockGuard(mutex);
		--activeThreads;
	}
	//std::cout << "Thread: " << thredID << " " << 100 << "%\n";
}
void xorBenchmark(size_t maxIteration, BenchmarkData& data)
{
	
	

	
	MultiSignalVector trainigsSet(4, 2);
	MultiSignalVector resultSet(4, 1);

	

	trainigsSet[0] = SignalVector(vector<float>{ 0, 0 });
	trainigsSet[1] = SignalVector(vector<float>{ 0, 1 });
	trainigsSet[2] = SignalVector(vector<float>{ 1, 0 });
	trainigsSet[3] = SignalVector(vector<float>{ 1, 1 });

	resultSet[0] = SignalVector(vector<float>{ 0 });
	resultSet[1] = SignalVector(vector<float>{ 1 });
	resultSet[2] = SignalVector(vector<float>{ 1 });
	resultSet[3] = SignalVector(vector<float>{ 0 });
	

	Display* display = nullptr;
	NetModel* netModel1 = nullptr;
	NetModel* netModel2 = nullptr;
	NetModel* netModel3 = nullptr;
	NetModel* netModel4 = nullptr;
	if (data.displayEnable)
	{
		display = new Display(sf::Vector2u(1900, 900), "X-OR Example");
		display->frameRateTarget(30);
	}

	
	double learnTime = 0;
	double minLearnTime = 9999;
	double maxLearnTime = 0;
	double averageLearnTime = 0;
	double timeout = 5000;

	for(size_t benchIt = 0; benchIt < maxIteration; ++benchIt)
	{
		
		if (data.enableDebugOutput)
			cout << "Start Benchmark Test [" << benchIt << "] ";
		float averageError = 0;
		float currentError = 0;
		size_t iteration = 0;

		Debug::Timer timer;

		BackpropNet net;
		net.setDimensions(2, data.net_xSize, data.net_ySize, 1);
		net.setStreamSize(trainigsSet.size());
		net.setActivation(Activation::sigmoid);
		net.setHardware(Hardware::gpu_cuda);
		net.setLearnParameter(1);
		net.build();

		if (data.displayEnable)
		{
			sf::Vector2f displaySize(display->getSize());

			sf::Vector2f spacing(60, 6);

			sf::Vector2f gridStartPos(100,100);
			sf::Vector2f gridSpacing(900, 400);
			float neuronSize = 5;

			netModel1 = new NetModel(&net);
			netModel1->setStreamIndex(0);
			netModel1->setNeuronSize(neuronSize);
			netModel1->setPos(gridStartPos);
			netModel1->setNeuronSpacing(spacing);
			display->addDrawable(netModel1);

			netModel2 = new NetModel(&net);
			netModel2->setStreamIndex(1);
			netModel2->setNeuronSize(neuronSize);
			netModel2->setPos(sf::Vector2f(gridStartPos.x, gridStartPos.y + gridSpacing.y));
			netModel2->setNeuronSpacing(spacing);
			display->addDrawable(netModel2);

			netModel3 = new NetModel(&net);
			netModel3->setStreamIndex(2);
			netModel3->setNeuronSize(neuronSize);
			netModel3->setPos(sf::Vector2f(gridStartPos.x + gridSpacing.x, gridStartPos.y));
			netModel3->setNeuronSpacing(spacing);
			display->addDrawable(netModel3);

			netModel4 = new NetModel(&net);
			netModel4->setStreamIndex(3);
			netModel4->setNeuronSize(neuronSize);
			netModel4->setPos(gridStartPos + gridSpacing);
			netModel4->setNeuronSpacing(spacing);
			display->addDrawable(netModel4);
		}

		
		learnTime = 0;
		timer.start();
		do{
			++iteration;

			timer.unpause();
			net.setInputVector(trainigsSet);
			net.calculate();
			net.learn(resultSet);
			timer.pause();

			if (data.displayEnable)
			if (display->needsFrameUpdate())
			{
				display->processEvents();
				display->draw();
			}

			MultiSignalVector err = net.getError();
			currentError = err.getRootMeanSquare();


			/*if (averageError < 0.1)
			{
				std::cout << "iteration [" << iteration << "]\t Error: " << averageError << "\n";
				net.setInputVector(trainigsSet);
				net.calculate();

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

			}*/
		} while (currentError > 0.05 && 
				 timeout > timer.getMillis());
		timer.stop();
		learnTime = timer.getMillis();
		if (minLearnTime > learnTime)
			minLearnTime = learnTime;
		else if (maxLearnTime < learnTime)
			maxLearnTime = learnTime;

		averageLearnTime += learnTime;

		if (data.displayEnable)
		{
			display->clearDrawable();
			delete netModel1;
			delete netModel2;
			delete netModel3;
			delete netModel4;

			netModel1 = nullptr;
			netModel2 = nullptr;
			netModel3 = nullptr;
			netModel4 = nullptr;
		}
		
		if(data.enableDebugOutput)
			cout << "End, Learn time: "<< learnTime << "ms\n";

		if (data.displayEnable)
		if (!display->isOpen())
			break;
	}
	averageLearnTime /= (float)maxIteration;
	if (data.enableDebugOutput)
	{
		cout << "Bench Finished\n";
		cout << "  Min learn Time:        " << minLearnTime << "ms\n";
		cout << "  Max learn Time:        " << maxLearnTime << "ms\n";
		cout << "  Average learn Time:    " << averageLearnTime << "ms\n";
	}
	

	data.averageLearnTime = averageLearnTime;
	data.minLearnTime = minLearnTime;
	data.maxLearnTime = maxLearnTime;

	if(display)
	delete display;
}

void printWeights(Net* net)
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


