#include "../inc/net.h"
#include <string>
#include <vector>
#include <SFML/Graphics.hpp>

using std::vector;
using std::cout;

sf::RenderWindow* window;

void printSignal(const SignalVector& sig);

void printBytes(void* ptr, size_t byteCount)
{
	std::cout << "Value @ " << (size_t)ptr;
	for (size_t i = 0; i < byteCount; ++i)
	{
		std::cout << " " << *((short*)ptr);
		ptr = (void*)((size_t)ptr+1);
	}
	std::cout << "\n";
}

void memcopyTest()
{
	struct Storage
	{
		unsigned short a;
		unsigned short b;
	};
	std::cout << "size of short: " << sizeof(short) <<"\n";
	std::cout << "size of Storage: " << sizeof(Storage) << "\n";

	unsigned short test[2];
	test[0] = 0xffff;
	test[1] = 0xffff;

	Storage *stor = (Storage*)&test;
	printBytes((void*)&stor, sizeof(Storage));

	Storage stor2;
	stor2 = *stor;
	printBytes((void*)&stor, sizeof(Storage));

}

void matrixProblemSolver(int width,int height);
void matrixTransposeExample(int width);
vector<double> transposeTimeListGPU1;
vector<double> transposeTimeListGPU2;
vector<double> transposeTimeListCPU;
vector<size_t> transposeWidthList;

int main(void)
{
	nvtxNameOsThread(1,"Main Thread");
	nvtxRangePush(__FUNCTION__);
	window = new sf::RenderWindow(sf::VideoMode(800, 600), "My window");
	//matrixProblemSolver(4, 3);

	/*for (int y = 2; y < 100; ++y)
	{
		matrixProblemSolver(10, y);

	}
	cout << "next\n";
	getchar();*/
	/*for (int y = 2; y < 100; ++y)
	{
		matrixProblemSolver(y, y);

	}
	cout << "next\n";
	getchar();*/
	//matrixProblemSolver(100, 2000);
	//matrixTransposeExample(2080);
	//return 0;
	//getchar();
	for (size_t i = 32; i < 1024*50; i+=1024)
	{
		matrixTransposeExample(i);
	}
	cout << "stats:\nwidth;GPU time 1 ms;GPU time 2 ms;CPU time ms\n";
	for (size_t i = 0; i < transposeTimeListGPU1.size(); ++i)
	{
		cout << transposeWidthList[i] << ";" << transposeTimeListGPU1[i]<<";"<< transposeTimeListGPU2[i] << ";"<<transposeTimeListCPU[i] << "\n";
	}
	cout << "finished\n";
	getchar();
	for (int x = 2; x < 100; ++x)
	{
		for (int y = 2; y < 100; ++y)
		{
			matrixProblemSolver(x, y);
			
		}
	}
	
	cout << "finished\n";
	getchar();
	window->close();
	return 0;
	{
		//NeuronalNet::GPU_CUDA_getSpecs();
		//memcopyTest();
		//for(int i=0; i<10;i++)
		//	NeuronalNet::GPU_CUDA_memcpyTest();
		//return 0;
		Net net;
		//cudaDeviceProp h_deviceProp;
		//cudaGetDeviceProperties(&h_deviceProp, 0);

		net.setDimensions(3, 100, 4000, 5);
		net.setActivation(Activation::sigmoid);
		net.setHardware(Hardware::cpu);
		net.build();

		net.setInputVector(SignalVector{ 1,1,1 });
		//net.calculate();
		net.calculate();
		SignalVector result = net.getOutputVector();
		printSignal(result);


		net.setHardware(Hardware::gpu_cuda);
		//net.calculate();
		//net.calculate();
		//net.calculate();
		net.calculate();
		net.calculate();
		result = net.getOutputVector();
		printSignal(result);
	}

	std::cout << "exiting";
	nvtxRangePop();
	return 0;
}


void printSignal(const SignalVector& sig)
{
	std::cout << "Signals: ";
	for (size_t i = 0; i < sig.size(); ++i)
	{
		printf("%f\t", sig[i]);
	}
	std::cout << "\n";
}

typedef vector<vector< int> > Matrix;

Matrix genMatrix(int w, int h);
Matrix getTransformed(const Matrix& m);
void printMatrix(const Matrix& m);
void plotMatrix(const Matrix& a, const Matrix& b);
int getElement(const Matrix& m, int index);
void calcAlgorithmus(const Matrix& m);
void convertLayerWeightToGPUWeight_getNewIndex(size_t startIndex, size_t& endIndex, size_t signalCount, size_t neuronCount);


void matrixProblemSolver(int width, int height)
{
	cout << "----------------------------\n";
	Matrix matrix = genMatrix(width, height);
	//printMatrix(matrix);
	//Matrix matrix2 = getTransformed(matrix);
	//printMatrix(matrix2);
	//plotMatrix(matrix, matrix2);
	calcAlgorithmus(matrix);
}

Matrix genMatrix(int w, int h)
{
	Matrix matrix(w, vector<int>(h));

	for (int x = 0; x < w; ++x)
	{
		for (int y = 0; y < h; ++y)
		{
			matrix[x][y] = y * w + x;
		}
	}
	return matrix;
}
Matrix getTransformed(const Matrix& m)
{
	int w = m.size();
	if (w == 0)
		return genMatrix(w,0);

	int h = m[0].size();
	Matrix result = genMatrix(w, h);

	for (int x = 0; x < w; ++x)
	{
		for (int y = 0; y < h; ++y)
		{
			int inputIndex = y * w + x;
			int outputIndex = inputIndex;
			result[(outputIndex) / h][outputIndex % h] = m[x][y];
		}
	}
	return result;
}

void printMatrix(const Matrix& m)
{
	int w = m.size();
	if (w == 0)
		return;

	int h = m[0].size();

	cout << "Matrix: width  = " << w << "\n";
	cout << "        height = " << h << "\n";
	for (int y = 0; y < h; ++y)
	{
		for (int x = 0; x < w; ++x)
		{
			printf("  %3i", m[x][y]);
		}
		cout << "\n\n";
	}
}

void plotMatrix(const Matrix& a, const Matrix& b)
{
	int w = a.size();
	if (w == 0)
		return;

	int h = a[0].size();
	int indexA = 0;
	int indexB = 0;


	int elements = w * h;
	/*for (int x = 0; x < elements; ++x)
	{
		cout << getElement(a, x)<<" ";
	}*/

	std::string txt = " ";
	cout << "\n   ";
	for (int x = 0; x < elements; ++x)
	{
		printf("%2i ", x);
	}
	cout << "\n";
	for (int y = 0; y < elements; ++y)
	{
		printf("%2i ", y);
		for (int x = 0; x < elements; ++x)
		{
			int v1 = getElement(a, y);
			int v2 = getElement(b, x);
			txt = "   ";
			if (v1 == v2)
				txt = " X ";

			cout << txt;
		}
		cout << "\n";
	}
}

int getElement(const Matrix& m, int index)
{
	int w = m.size();
	if (w == 0)
		return 0;

	int h = m[0].size();

	int x = index % w;
	int y = index / w;
	return m[x][y];
}


void calcAlgorithmus(const Matrix& m) 
{

	float* d_list;
	size_t signalCount = m.size();
	size_t neuronCount = m[0].size();
	cout << "calcAlgorithmus: "<< signalCount << " x "<< neuronCount<<"\n";

	//----------------------

	size_t weightCount = signalCount * neuronCount;
	size_t circuitsMemSize = 10;
	short *circuitStartIndex = new short[circuitsMemSize];
	short*circuitNodeCount  = new short[circuitsMemSize];

	memset(circuitStartIndex, 0, circuitsMemSize * sizeof(short));
	memset(circuitNodeCount,  0, circuitsMemSize * sizeof(short));

	size_t circuitIndex = 0;

	struct Point
	{
		float x;
		float y;
	};
	vector<Point> points;
	vector<vector<Point> >circuits;

	// Kreisläufe finden
	// Erstes und letztes Elemnt überspringen, diese bleiben konstant.
	for (size_t currentStartIndex = 1; currentStartIndex < weightCount - 1; ++currentStartIndex)
	{
		size_t destinationIndex = 0;
		size_t srcIndex = currentStartIndex;
		bool circuitAlreadyExists = false;
		size_t nodeCount = 0;
		points.clear();
		cout << "index: " << currentStartIndex << "\n";
		while (destinationIndex != currentStartIndex)
		{
			

			//cout << "startIndex: " << srcIndex << " destination: " << destinationIndex << "\n";
			convertLayerWeightToGPUWeight_getNewIndex(srcIndex, destinationIndex, signalCount, neuronCount);
			
			for (int i = 0; i < 5; ++i)
			{
				if (circuitStartIndex[i] == destinationIndex)
				{
					
					circuitAlreadyExists = true;
				}
			}
			Point p;
			p.x = destinationIndex;
			p.y = srcIndex;
			points.push_back(p);

			++nodeCount;
			
			srcIndex = destinationIndex;
		}

		if (!circuitAlreadyExists)
		{
			if (circuitIndex > circuitsMemSize)
			{
				//printf("ERROR: more than 2 circuits found\n");
				cout<<"realocating memory: "<< circuitsMemSize * 2<<"\n";

				short* newBuff = new short[circuitsMemSize*3];
				memcpy(newBuff, circuitStartIndex, circuitsMemSize * sizeof(short));
				memset(&newBuff[circuitsMemSize], 0, circuitsMemSize*3);
				delete[] circuitStartIndex;
				circuitStartIndex = newBuff;

				newBuff = new short[circuitsMemSize * 3];
				memcpy(newBuff, circuitNodeCount, circuitsMemSize * sizeof(short));
				memset(&newBuff[circuitsMemSize], 0, circuitsMemSize*3);
				delete[] circuitNodeCount;
				circuitNodeCount = newBuff;

				circuitsMemSize *= 3;
			}
			circuits.push_back(points);
			circuitNodeCount[circuitIndex] = nodeCount;
			circuitStartIndex[circuitIndex] = currentStartIndex;
			++circuitIndex;
			
		}
	}
	cout << "Circuits: " << circuitIndex<<"\n";

	for (size_t i = 0; i < circuitIndex; ++i)
	{
		//cout << "  Start index: " << circuitStartIndex[i] << "\tNodecount: "<<circuitNodeCount[i] <<"\n";
	}

	cout << "WindowUpdate\n";

	window->clear();
	sf::Vector2f scale(500 /(float)weightCount, 500 / (float)weightCount);
	sf::CircleShape shape1(scale.x * 1.f);
	shape1.setOrigin(shape1.getRadius(), shape1.getRadius());
	shape1.setPosition(sf::Vector2f(0* scale.x, 0* scale.y));
	shape1.setFillColor(sf::Color(255,0,0));
	window->draw(shape1);
	sf::CircleShape shape2(scale.x * 1.f);
	shape2.setOrigin(shape2.getRadius()*2, shape2.getRadius()*2);
	shape2.setPosition(sf::Vector2f(weightCount* scale.x, weightCount* scale.y));
	shape2.setFillColor(sf::Color(255, 0, 0));
	window->draw(shape2);
	for (size_t i = 0; i < circuits.size(); ++i)
	{
		sf::Color col = sf::Color(rand() % 255, rand() % 255, rand() % 255);
		if (circuits[i].size() == 1)
		{
			sf::CircleShape shape(scale.x*1.f);
			shape.setOrigin(shape.getRadius(), shape.getRadius());
			shape.setPosition(sf::Vector2f(circuits[i][0].x* scale.x, circuits[i][0].y* scale.y));
			shape.setFillColor(col);
			window->draw(shape);
		}
		else
		{
			sf::Vertex* strip = new sf::Vertex[circuits[i].size() + 1];
			
			for (size_t j = 0; j < circuits[i].size(); ++j)
			{
				strip[j] = sf::Vertex(sf::Vector2f(circuits[i][j].x * scale.x, circuits[i][j].y * scale.y));
				strip[j].position += sf::Vector2f((float)(rand() % 200) / 100-1, (float)(rand() % 200) / 100-1);
				strip[j].color = col;
			}
			strip[circuits[i].size()] = sf::Vertex(sf::Vector2f(circuits[i][0].x * scale.x, circuits[i][0].y * scale.y));
			strip[circuits[i].size()].color = col;
			window->draw(strip, circuits[i].size() + 1, sf::LinesStrip);
			delete[] strip;
		}
		
	}
	cout << "display\n";
	window->display();
	cout << "display done\n";
	while (window->isOpen())
	{
		//cout << "event1\n";
		sf::Event event;
		while (window->pollEvent(event))
		{
			//cout << "event2\n";
			// "close requested" event: we close the window
			if (event.type == sf::Event::Closed)
				window->close();
			if (event.type == sf::Event::KeyPressed)
			{
				if (event.key.code == sf::Keyboard::Enter)
				{
					cout << "return\n";
					return;
				}
			}
			
		}
		//cout << "return\n";
		//return;
	}
	cout << "return\n";
}
void convertLayerWeightToGPUWeight_getNewIndex(size_t startIndex, size_t& endIndex, size_t signalCount, size_t neuronCount)
{
	// Calculates the new Position of a element
	endIndex = startIndex / neuronCount + (startIndex % neuronCount) * signalCount;
}

void plotMatrix1(float* list, size_t width)
{
	cout << "Matrix: \n";
	for (size_t y = 0; y < width; ++y)
	{
		for (size_t x = 0; x < width; ++x)
		{
			printf("%1.0f", list[y * width + x]);
		}
		cout << "\n";
	}
}

void matrixTransposeCPU(float* list, size_t width)
{
	for (int y = 0; y < width; ++y)
		for (int x = 0; x < width; ++x) {
			if (x > y)
				continue;
			float tmp = list[y * width + x];
			list[y * width + x] = list[y + width * x];
			list[y + width * x] = tmp;
		}
}
bool matrixComp(float* org, float* tst, size_t width)
{
	size_t size = width * width;
	for (size_t i = 0; i < size; ++i)
		if (org[i] != tst[i])
		{
			//cout << "-not equal: elem: " << i << " "<<org[i] <<" != "<<tst[i]<<"\n";
			return false;
		}
	return true;
}

double matrixTransposeSelfTest(float *h_list,float *h_original, size_t width,size_t iterations, bool transposeCheck)
{
	cout << "  matrixTransposeSelfTest begin\n";
	float* d_list;
	size_t elementCount = width * width;
	cout << "    Alloc GPU Memory... ";
	NeuronalNet::GPU_CUDA_allocMem(d_list, elementCount * sizeof(float));
	cout << "done\n";
	cout << "    Transfer GPU Memory to device... ";
	NeuronalNet::GPU_CUDA_transferToDevice(d_list, h_list, elementCount * sizeof(float));
	cout << "done\n";

	double time = 0;
	for (size_t i = 0; i < iterations; ++i)
	{
		time += NeuronalNet::GPU_CUDA_transposeMatrix(d_list, width);
		if (transposeCheck)
		{
			NeuronalNet::GPU_CUDA_transferToHost(d_list, h_list, elementCount * sizeof(float));
			if (i % 2 == 0)
			{
				if (matrixComp(h_original, h_list, width))
				{
					cout << "  !!! NOT TRANSPOSED 1 !!! " << width << "\n";
					plotMatrix1(h_list, width);
					getchar();
				}
			}
			else
			{
				if (!matrixComp(h_original, h_list, width))
				{
					cout << "  !!! NOT EQUAL 1 !!! " << width << "\n";
					//plotMatrix1(matrixORG, width);
					plotMatrix1(h_list, width);
					getchar();
				}
			}
		}
	}

	cout << "    Transfer GPU Memory to host... ";
	NeuronalNet::GPU_CUDA_transferToHost(d_list, h_list, elementCount * sizeof(float));
	cout << "done\n";
	cout << "    Delete GPU Memory... ";
	NeuronalNet::GPU_CUDA_freeMem(d_list);
	cout << "done\n";
	cout << "  matrixTransposeSelfTest end \n";
	return time / iterations;
}
double matrixTransposeCudaExampleTest(float* h_list, float* h_original, size_t width, size_t iterations, bool transposeCheck)
{
	cout << "  matrixTransposeCudaExampleTest begin\n";
	size_t elementCount = width * width;
	float* d_list1;
	float* d_list2;
	cout << "    Alloc GPU Memory... ";
	NeuronalNet::GPU_CUDA_allocMem(d_list1, elementCount * sizeof(float));
	NeuronalNet::GPU_CUDA_allocMem(d_list2, elementCount * sizeof(float));
	cout << "done\n";
	cout << "    Transfer GPU Memory to device... ";
	NeuronalNet::GPU_CUDA_transferToDevice(d_list1, h_list, elementCount * sizeof(float));
	cout << "done\n";

	double time = 0;
	for (size_t i = 0; i < iterations; ++i)
	{
		time += NeuronalNet::GPU_CUDA_transposeMatrix2(d_list1,d_list2, width);
		if (transposeCheck)
		{
			NeuronalNet::GPU_CUDA_transferToHost(d_list2, h_list, elementCount * sizeof(float));
			if (i % 2 == 0)
			{
				if (matrixComp(h_original, h_list, width))
				{
					cout << "  !!! NOT TRANSPOSED 2 !!! " << width << "\n";
					plotMatrix1(h_list, width);
					getchar();
				}
			}
			else
			{
				if (!matrixComp(h_original, h_list, width))
				{
					cout << "  !!! NOT EQUAL 2 !!! " << width << "\n";
					//plotMatrix1(matrixORG, width);
					plotMatrix1(h_list, width);
					getchar();
				}
			}
		}
		float* tmp = d_list1;
		d_list1 = d_list2;
		d_list2 = tmp;
	}
	cout << "    Transfer GPU Memory to host... ";
	NeuronalNet::GPU_CUDA_transferToHost(d_list2, h_list, elementCount * sizeof(float));
	cout << "done\n";
	cout << "    Delete GPU Memory... ";
	NeuronalNet::GPU_CUDA_freeMem(d_list1);
	NeuronalNet::GPU_CUDA_freeMem(d_list2);
	cout << "done\n";
	cout << "  matrixTransposeCudaExampleTest end\n";
	return time / iterations;
}

void plotResults(size_t width, double timeGPU_self, double timeGPU_cudaExample, double timeCPU)
{
	cout << "    Plot results... ";
	static bool firstCall = true;
	FILE* file;
	if (firstCall)
	{
		file = fopen("result.csv", "w");
	}
	else
	{
		file = fopen("result.csv", "a");
	}
	
	if (file)
	{
		if (firstCall)
		{
			firstCall = false;
			fprintf(file, "width;time GPU self ms;time GPU cuda example ms;time CPU ms\n");
		}
		fprintf(file,"%lu;%lf;%lf;%lf\n", width, timeGPU_self, timeGPU_cudaExample, timeCPU);
		fclose(file);
	}
	else
	{
		cout << " ERROR: can't open file \n";
		return;
	}
	cout << "done\n";
}
void matrixTransposeExample(int width)
{
	int iterations = 4;
	bool validateTransposition = false;

	bool calcCudaSelf = true;
	bool calcCudaExample = true;
	bool calcCPU = true;
	double cudaMaxMem = 7e+09;


	double bytes = width * width * sizeof(float);
	cout << "Transpose Test, width: " << width << " " << bytes << "bytes " << bytes / 1000 << "KB " << bytes / 1000000 << "MB\n";


	cout << "  Alloc CPU Memory... ";
	float* matrixORG;
	if (validateTransposition)
		matrixORG = new float[width * width];
	float* matrix = new float[width * width];
	cout << "done\n";

	if (validateTransposition)
	{
		cout << "  Fill matrix... ";
		for (size_t x = 0; x < width; ++x)
		{
			for (size_t y = 0; y < width; ++y)
			{
				//matrix[y * width + x] = (float)(rand() % 10);
				matrix[y * width + x] = (rand() % 100) / 10.f;
				/*if (y > x)
					matrixORG[y * width + x] = 0;
				else
					matrixORG[y * width + x] = 1;//NeuronalNet::gaussSum(x)+y;*/
			}
		}
		cout << "done\n";
	

	
		memcpy(matrixORG, matrix, width * width * sizeof(float));

		if (!matrixComp(matrixORG, matrix, width))
		{
			cout << "  !!! NOT EQUAL 0 !!! " << width << "\n";
			//plotMatrix1(matrixORG, width);
			plotMatrix1(matrix, width);
			getchar();
		}
	}
	//plotMatrix1(matrix, width);
	//cout << "transpose...\n";
	
	double timeMs1 = 0;
	if (calcCudaSelf )
	{
		if (bytes > cudaMaxMem)
		{
			cout << "  Skip matrixTransposeSelfTest, out of memory: " << bytes / 1000000 << "MB, max is: " << cudaMaxMem / 1000000 << "MB\n";
		}
		else
			timeMs1 = matrixTransposeSelfTest(matrix, matrixORG, width, iterations, validateTransposition);
	}
	/*double timeMs1 = 0;
	for (int i = 0; i < iterations; ++i)
	{
		timeMs1 += NeuronalNet::GPU_CUDA_transposeMatrix(matrix, width);
		if (i % 2 == 0)
		{
			if (matrixComp(matrixORG, matrix, width))
			{
				cout << "  !!! NOT TRANSPOSED 1 !!! " << width << "\n";
				plotMatrix1(matrix, width);
				getchar();
			}
		}
		else
		{
			if (!matrixComp(matrixORG, matrix, width))
			{
				cout << "  !!! NOT EQUAL 1 !!! " << width << "\n";
				//plotMatrix1(matrixORG, width);
				plotMatrix1(matrix, width);
				getchar();
			}
		}
	}
	timeMs1 /= iterations;*/
	if(validateTransposition)
		if (!matrixComp(matrixORG, matrix, width))
		{
			cout << "  !!! NOT EQUAL 0.1 !!! " << width << "\n";
			//plotMatrix1(matrixORG, width);
			plotMatrix1(matrix, width);
			getchar();
		}

	//-------------------------------------
	double timeMs2 = 0;
	if (calcCudaExample)
	{
		if (bytes > cudaMaxMem/2)
		{
			cout << "  Skip matrixTransposeCudaExampleTest, out of memory: " << bytes / 1000000 << "MB, max is: " << cudaMaxMem / 2000000 << "MB\n";
		}
		else
			timeMs2 = matrixTransposeCudaExampleTest(matrix, matrixORG, width, iterations, validateTransposition);
	}
		
	/*double timeMs2 = 0;
	for (int i = 0; i < iterations; ++i)
	{
		timeMs2 += NeuronalNet::GPU_CUDA_transposeMatrix2(matrix, width);
		if (i % 2 == 0)
		{
			if (matrixComp(matrixORG, matrix, width))
			{
				cout << "  !!! NOT TRANSPOSED 2 !!! " << width << "\n";
				plotMatrix1(matrix, width);
				getchar();
			}
		}
		else
		{
			if (!matrixComp(matrixORG, matrix, width))
			{
				cout << "  !!! NOT EQUAL 2 !!! " << width << "\n";
				//plotMatrix1(matrixORG, width);
				plotMatrix1(matrix, width);
				getchar();
			}
		}
	}
	timeMs2 /= iterations;*/
	
	
 
	//cout << "transpose done\n";
	//plotMatrix1(matrix, width);


	//cout << "transpose...\n";
	double transposeCPUTimeMs = 0;
	if (calcCPU)
	{
		if (bytes > 30e+09)
		{
			cout << "  Skip CPU matrix transpose test, out of memory: " << bytes / 1000000 << "MB, max is: " << 30000 << "MB\n";
		}
		else
		{
			cout << "  CPU matrix transpose begin\n";
			auto t1 = std::chrono::high_resolution_clock::now();
			matrixTransposeCPU(matrix, width);
			auto t2 = std::chrono::high_resolution_clock::now();
			transposeCPUTimeMs = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1000000.f;
			cout << "  CPU matrix transpose end\n";
		}
	}
	//cout << "transpose done\n";
	//plotMatrix1(matrix, width);

	cout << "  Average CUDA Transpose time:         " << timeMs1 << "ms\n";
	cout << "  Average CUDA Example Transpose time: " << timeMs2 << "ms\n";
	cout << "  Average CPU Transpose time:          " << transposeCPUTimeMs << "ms\n";
	cout << "  Delete CPU Memory... ";
	delete[] matrix;
	if (validateTransposition)
		delete[] matrixORG;
	cout << "done\n";
	transposeTimeListGPU1.push_back(timeMs1);
	transposeTimeListGPU2.push_back(timeMs2);
	transposeTimeListCPU.push_back(transposeCPUTimeMs);
	transposeWidthList.push_back(width);
	plotResults(width, timeMs1, timeMs2, transposeCPUTimeMs);
	cout << "Transpose Test, width: " << width << " done\n\n";
	//Sleep(10);
	//getchar();
}