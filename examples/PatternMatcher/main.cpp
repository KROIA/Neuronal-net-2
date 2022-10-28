#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include "neuronalNet.h"

using std::vector;
using std::cout;
using std::string;
using namespace NeuronalNet;
using namespace Graphics;

MultiSignalVector trainingInputs;
MultiSignalVector trainingOutputs;
MultiSignalVector testInputs;
MultiSignalVector testOutputs;

float normalScale = 10;
float outputOffset = -1;

void readData();


int main()
{
	readData();
	Display display(sf::Vector2u(800, 800), "PatternMatching Example");
	BackpropNet net;
	net.setDimensions(2, 2, 15, 1);
	net.setActivation(Activation::sigmoid);
	net.setHardware(Hardware::cpu);
	net.setLearnParameter(0.5);
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


	float error = 1;
	size_t counter = 0;
	while (error > 0.01)
	{
		net.setLearnParameter(error);
		error = 0;
		for (size_t i = 0; i < trainingInputs.size(); ++i)
		{
			net.setInputVector(trainingInputs[i]);
			net.calculate();
			net.learn(trainingOutputs[i]);
			error += net.getError().getRootMeanSquare();

			if (counter == 0 || counter % 10000 == 0)
			{
				SignalVector result = net.getOutputVector();

				cout << "Set: " << i << "\tInputs: ";
				for (size_t j = 0; j < trainingInputs[i].size(); ++j)
				{
					cout << trainingInputs[i][j] << ",\t";
				}
				cout << " Target: ";
				for (size_t j = 0; j < trainingOutputs[i].size(); ++j)
				{
					cout << trainingOutputs[i][j] << ",\t";
				}
				cout << "\tPredicted: ";
				for (size_t j = 0; j < result.size(); ++j)
				{
					cout << result[j] << ", ";
				}

				cout << string(abs(trainingOutputs[i][0] - result[0]) * 10.f, '#') << "\n";
				//display.processEvents();
				//display.draw();
			}
			

		}
		display.processEvents();
		display.draw();
		if (counter == 0 || counter % 10000 == 0)
		{
			const float* w;
			w = net.getWeight();
			cout << "w: ";
			char str[10];
			for (size_t i = 0; i < net.getWeightSize(); ++i)
			{
				sprintf_s(str, "%2.3f", *(w+i));
				cout << str << "  ";
			}
			cout << "\n";
				
			getchar();
			// Validate
			cout << "Validate\n";
			struct Point
			{
				float x;
				float y;
			};
			vector<Point> classZero;
			vector<Point> classOne;
			vector<Point> classTwo;
			for (size_t i = 0; i < testInputs.size(); ++i)
			{
				net.setInputVector(testInputs[i]);
				net.calculate();
				SignalVector result = net.getOutputVector();

				cout << "Set: " << i << "\tInputs: ";
				for (size_t j = 0; j < trainingInputs[i].size(); ++j)
				{
					cout << trainingInputs[i][j] << ",\t";
				}
				cout << " Target: ";
				for (size_t j = 0; j < testOutputs[i].size(); ++j)
				{
					cout << testOutputs[i][j] << ", ";
				}
				cout << "\tPredicted: ";
				for (size_t j = 0; j < result.size(); ++j)
				{
					cout << result[j] << ", ";
				}
				cout << string(abs(testOutputs[i][0] - result[0]) * 10.f, '#') << "\n";

				Point point;
				point.x = testInputs[i][0] * normalScale;
				point.y = testInputs[i][1] * normalScale;

				if (result[0] < -0.33)
				{
					classZero.push_back(point);
				}
				else if (result[0] < 0.33)
				{
					classOne.push_back(point);
				}
				else
				{
					classTwo.push_back(point);
				}


			}
			cout << "save File... ";

			FILE* pFile;
			fopen_s(&pFile,"result.csv", "w");
			vector<string> fileBuff;
			fileBuff.push_back("0 x;0 y;;1 x;1 y;;2 x;2 y\n");
			if (pFile)
			{
				//fprintf(pFile, "0 x;0 y\n");
				for (size_t i = 0; i < classZero.size(); ++i)
				{
					//fprintf(pFile, "%f;%f\n", classZero[i].x, classZero[i].y);
					if (fileBuff.size() <= i)
						fileBuff.push_back("");
					fileBuff[i] = fileBuff[i]  + std::to_string(classZero[i].x) + ";" + std::to_string(classZero[i].y) + ";;";
				}
				//fclose(pFile);
			}
			//fopen_s(&pFile, "result_1.csv", "w");
			if (pFile)
			{
				//fprintf(pFile, "1 x;1 y\n");
				for (size_t i = 0; i < classOne.size(); ++i)
				{
					//fprintf(pFile, "%f;%f\n", classOne[i].x, classOne[i].y);
					if (fileBuff.size() <= i)
						fileBuff.push_back(";;;");
					fileBuff[i] = fileBuff[i] + std::to_string(classOne[i].x) + ";" + std::to_string(classOne[i].y) + ";;";
				}
				//fclose(pFile);
			}
			//fopen_s(&pFile, "result_2.csv", "w");
			if (pFile)
			{
				//fprintf(pFile, "2 x;2 y\n");
				for (size_t i = 0; i < classTwo.size(); ++i)
				{
					//fprintf(pFile, "%f;%f\n", classTwo[i].x, classTwo[i].y);
					if (fileBuff.size() <= i)
						fileBuff.push_back(";;;;;");
					fileBuff[i] = fileBuff[i] + std::to_string(classTwo[i].x) + ";" + std::to_string(classTwo[i].y) + ";;";
				}
				//fclose(pFile);
			}
			if (pFile)
			{
				for (size_t i = 0; i < fileBuff.size(); ++i)
					fprintf(pFile, "%s\n", fileBuff[i].c_str());
				fclose(pFile);
			}
			getchar();
		}
		error /= (float)trainingInputs.size();
		++counter;
		if (counter % 50 == 0)
			cout << "Error: " << error << "\n";

	}

	// Validate
	for (size_t i = 0; i < testInputs.size(); ++i)
	{
		net.setInputVector(testInputs[i]);
		net.calculate();
		SignalVector result = net.getOutputVector();

		cout << "Set: " << i << " Target: ";
		for (size_t j = 0; j < testOutputs[i].size(); ++j)
		{
			cout << testOutputs[i][j] << ", ";
		}
		cout << "\tPredicted: ";
		for (size_t j = 0; j < result.size(); ++j)
		{
			cout << result[j] << ", ";
		}
		cout << string(abs(testOutputs[i][0] - result[0]) * 10.f, '#') << "\n";

		
	}
	const float* w;
	w = net.getWeight();
	cout << "w: ";
	char str[10];
	for (size_t i = 0; i < net.getWeightSize(); ++i)
	{
		sprintf_s(str, "%2.3f", *(w + i));
		cout << str << "  ";
	}
	cout << "\n";
}


void readData()
{
	trainingInputs.clear();
	trainingOutputs.clear();
	testInputs.clear();
	testOutputs.clear();

	/*string dataPaht = "E:\\Dokumente\\Visual Studio 2019\\Projects\\Neural-net-2\\examples\\PatternMatcher\\Data";
	string fileNameTrain = "KNN_Train.csv";
	string fileNameTest  = "KNN_Test.csv";

	std::ifstream file(dataPaht+"\\"+ fileNameTrain);
	vector<vector<float> > inpTrain;
	vector<vector<float> > expOut;
	bool firstLine = true;
	if (file.is_open()) {
		std::string line;
		while (std::getline(file, line)) {
			if (firstLine)
			{
				firstLine = false;
				continue;
			}
			inpTrain.push_back(vector<float>());
			while (line.find(",") != string::npos)
			{

			}

		}
		file.close();
	}*/

	trainingInputs = MultiSignalVector(vector<vector<float> >
	{
		{-0.725589736, 5.6279375    },
		{ -1.869544153  , 7.932427239 },
		{ 5.42820358    , 0.063394768 },
		{ 17.71247407   , 6.271720156 },
		{ 2.236148502   , 2.437073551 },
		{ 8.309924599   , -0.948964614 },
		{ -0.266973685  , 7.095013109 },
		{ 2.559625273   , 0.768681443 },
		{ -0.279290417  , 6.898579579 },
		{ 9.307548039   , 8.201943352 },
		{ 7.630846692   , -1.437901067 },
		{ -0.124105491  , 10.41442765 },
		{ 8.811076313   , -7.578650183 },
		{ 4.284423198   , -6.841834621 },
		{ 0.53594468    , -0.546020323 },
		{ 4.165343912   , 4.692045751 },
		{ 7.047946226   , -6.613236844 },
		{ 10.40222763   , 3.243296006 },
		{ -2.048856913  , 6.228490036 },
		{ 6.35830522    , -2.407407609 },
		{ 3.487753388   , 6.162886009 },
		{ 4.105234786   , -7.176629439 },
		{ 9.781760759   , 6.380033252 },
		{ 6.147364198   , -6.73554797 },
		{ 4.554643379   , 0.965290557 },
		{ 8.981164706   , -0.618415063 },
		{ 7.271275221   , -7.211356553 },
		{ -1.440470787  , 5.190514586 },
		{ 10.66804457   , 6.233201888 },
		{ 7.01467052    , -1.36601115 },
		{ 7.34144509    , -4.168633181 },
		{ 11.11262764   , -4.509536504 },
		{ 7.725116396   , 0.236620429 },
		{ 0.236484684   , 1.078417733 },
		{ 13.5404661    , 1.769897152 },
		{ 3.039994236   , 1.633781196 },
		{ 1.615980498   , -12.88105152 },
		{ 1.03328826    , 7.787791646 },
		{ 9.117308724   , -1.987289899 },
		{ 2.131416203   , 5.12574818 },
		{ 2.26463518    , -6.600126134 },
		{ 0.830966521   , -4.494050341 },
		{ 4.764516724   , 0.066110819 },
		{ 3.162875114   , 0.983831785 },
		{ 9.526059865   , -6.72513587 },
		{ 4.531695631   , 5.161474485 },
		{ 9.720071788   , 4.774266176 },
		{ 9.085407852   , 5.568598688 },
		{ 10.13522325   , 0.817028001 },
		{ 4.170110523   , -5.339738289 },
		{ 7.781496498   , 1.202892335 },
		{ 0.160073408   , -7.848686242 },
		{ 0.694709902   , -5.016973636 },
		{ 6.151592204   , -6.442990081 },
		{ 6.513960361   , -4.192041286 },
		{ 9.401097518   , 3.144996652 },
		{ 3.784225991   , 6.0515594 },
		{ 4.73259721    , -7.310557208 },
		{ 10.1931215    , -2.54092707 },
		{ 8.595881138   , 5.63286532 },
		{ 11.06015712   , 5.476354048 },
		{ 10.77788111   , 6.187601414 },
		{ -0.225901613  , -10.08901599 },
		{ -1.890377111  , -7.518602327 },
		{ 7.643172284   , -7.430276093 },
		{ 0.939626115   , 9.207009064 },
		{ 6.783318823   , -5.933185133 },
		{ -1.485681259  , 7.427594779 },
		{ 13.94930666   , 3.132916597 },
		{ -0.956796356  , -2.920353515 },
		{ 2.776932148   , -10.31597888 },
		{ 8.074656462   , -5.150689445 },
		{ -2.716111033  , 4.967239648 },
		{ 2.022936461   , -11.40025168 },
		{ 6.451365161   , 2.733244135 },
		{ -2.003295001  , 6.78471924 },
		{ 14.90446913   , 7.945782703 },
		{ 7.933658148   , -2.225139034 },
		{ 8.377592603   , 1.339916057 },
		{ 12.67436983   , 1.346517114 },
		{ 9.49163642    , 4.27825655 },
		{ -4.329213459  , -8.12689292 },
		{ 8.82972034    , -6.190955469 },
		{ 1.542533765   , 7.807303148 },
		{ 14.83338845   , -4.401135767 },
		{ 1.311391525   , 7.927850457 },
		{ 8.220669201   , 6.020936973 },
		{ 9.582059955   , -1.156882964 },
		{ 17.40507544   , 1.655594641 },
		{ 15.66034423   , 2.737902677 },
		{ 7.923252003   , 1.105157462 },
		{ 2.176965975   , 6.749829666 },
		{ 9.008637982   , 2.6731679 },
		{ 15.36126802   , 0.422393687 },
		{ 11.26223294   , -2.802165047 },
		{ -3.094370145  , 6.94664828 },
		{ 6.58386489    , -3.722363691 },
		{ 2.168021865   , -3.253257716 },
		{ 6.755212319   , 3.703095076 },
		{ 4.75310799    , -1.443065534 }
	});

	trainingOutputs = MultiSignalVector(vector<vector<float> >
	{
		{0 },
		{ 0 },
		{ 1 },
		{ 2 },
		{ 0 },
		{ 2 },
		{ 0 },
		{ 0 },
		{ 0 },
		{ 0 },
		{ 2 },
		{ 0 },
		{ 1 },
		{ 1 },
		{ 0 },
		{ 0 },
		{ 1 },
		{ 2 },
		{ 0 },
		{ 1 },
		{ 0 },
		{ 1 },
		{ 0 },
		{ 1 },
		{ 1 },
		{ 2 },
		{ 1 },
		{ 0 },
		{ 0 },
		{ 2 },
		{ 1 },
		{ 1 },
		{ 0 },
		{ 1 },
		{ 2 },
		{ 0 },
		{ 1 },
		{ 0 },
		{ 2 },
		{ 0 },
		{ 1 },
		{ 1 },
		{ 2 },
		{ 0 },
		{ 1 },
		{ 0 },
		{ 2 },
		{ 2 },
		{ 2 },
		{ 1 },
		{ 2 },
		{ 1 },
		{ 1 },
		{ 1 },
		{ 1 },
		{ 2 },
		{ 0 },
		{ 1 },
		{ 2 },
		{ 2 },
		{ 2 },
		{ 2 },
		{ 1 },
		{ 1 },
		{ 1 },
		{ 0 },
		{ 1 },
		{ 0 },
		{ 2 },
		{ 1 },
		{ 1 },
		{ 1 },
		{ 0 },
		{ 1 },
		{ 0 },
		{ 0 },
		{ 2 },
		{ 2 },
		{ 2 },
		{ 2 },
		{ 2 },
		{ 1 },
		{ 1 },
		{ 0 },
		{ 2 },
		{ 0 },
		{ 2 },
		{ 2 },
		{ 2 },
		{ 2 },
		{ 0 },
		{ 0 },
		{ 2 },
		{ 2 },
		{ 2 },
		{ 0 },
		{ 1 },
		{ 0 },
		{ 2 },
		{ 0 }

	});


	// Test Data

	testInputs = MultiSignalVector(vector<vector<float> >
	{
		{0.151398867, 1.244609822     },
		{ 9.613435408    , 0.339232777 },
		{ 11.82201718    , -1.505194864 },
		{ 3.366778943    , -8.113722197 },
		{ 17.66744905    , -4.13726323 },
		{ 12.77482403    , 0.317642446 },
		{ 3.068235229    , -5.885849087 },
		{ 4.426374448    , -2.046806322 },
		{ 2.147748114    , 7.039853248 },
		{ 10.87079706    , 1.461755967 },
		{ 14.06118499    , 3.793224507 },
		{ 1.328575567    , 1.893298626 },
		{ 10.06093293    , -1.427065551 },
		{ 10.15284148    , 1.707900151 },
		{ 11.64951392    , -10.88773717 },
		{ 10.81373096    , 4.888929924 },
		{ -0.420812631   , 3.611362656 },
		{ 11.76114685    , 7.026290589 },
		{ 12.80651361    , 2.662985544 },
		{ 5.546044357    , 3.597417022 },
		{ 11.73323963    , 1.888538941 },
		{ 3.795254371    , 3.956675077 },
		{ 8.106841175    , 2.335775847 },
		{ 11.85312864    , 2.981954235 },
		{ 10.27237395    , 4.664330917 },
		{ 2.928285956    , -3.102088751 },
		{ 2.420556334    , 7.160210884 },
		{ 3.116636836    , 4.127203686 },
		{ -0.305203475   , -4.073118764 },
		{ 1.569884211    , 1.72480698 },
		{ 3.05059612     , 3.767125537 },
		{ -0.805910621   , -4.896942128 },
		{ 7.80930762     , 7.597154562 },
		{ 5.226910925    , -10.59299194 },
		{ 8.19621258     , -5.351952539 },
		{ 6.961562267    , -9.008305793 },
		{ 6.591947579    , -9.337952106 },
		{ 3.913840654    , -8.713065667 },
		{ 13.63588611    , 10.15442876 },
		{ 1.652681998    , -4.678526748 },
		{ 5.482230239    , -5.729309761 },
		{ 7.11737778     , -2.210492483 },
		{ -0.197438113   , -2.978291389 },
		{ 4.652964868    , -4.564819235 },
		{ 5.412220746    , -3.922475169 },
		{ -1.85216712    , 9.006532984 },
		{ 5.960702856    , 10.31935886 },
		{ -0.349361458   , 4.955673904 },
		{ 5.152995822    , 2.606952493 },
		{ 4.805331574    , -3.332960869 },
		{ 11.54477555    , 2.149456684 },
		{ 13.04110155    , 5.12644074 },
		{ 12.30667429    , -3.196034192 },
		{ 2.668204196    , 12.35089298 },
		{ 8.190169397    , -0.989593897 },
		{ 2.796512136    , -5.847411001 },
		{ -2.502936526   , -1.673425893 },
		{ 1.396501274    , 1.927495537 },
		{ -0.640392793   , 9.385038543 },
		{ 3.221851501    , -6.702653241 },
		{ 13.45217316    , 3.247368655 },
		{ 12.10811748    , 3.970987713 },
		{ 7.924961979    , -4.568848117 },
		{ 8.458145756    , 0.884856122 },
		{ 18.84913997    , 1.629778278 },
		{ 2.999555638    , 8.097994075 },
		{ 14.66009989    , 1.494422478 },
		{ 5.968193048    , 1.150380564 },
		{ 12.83833085    , 6.155547331 },
		{ 2.435050023    , -1.714223033 },
		{ 0.16780854     , -5.407161484 },
		{ 16.66630548    , 6.388698445 },
		{ 1.268870183    , 7.053181169 },
		{ -1.91058622    , 5.820454942 },
		{ 4.560129743    , -2.933408776 },
		{ 9.347005798    , 2.143889446 },
		{ 7.735791591    , 5.655204121 },
		{ 7.685208773    , 2.858852552 },
		{ 1.829997046    , -3.556887881 },
		{ 3.709813513    , 4.735499004 },
		{ 9.696777636    , -4.816959062 },
		{ -1.054398319   , -7.76563052 },
		{ 10.98232777    , 2.244105537 },
		{ 1.062391721    , 6.856244134 },
		{ 0.871539786    , 3.655809759 },
		{ 2.177827142    , 2.770515237 },
		{ 5.155567522    , 6.758730186 },
		{ 5.104722315    , -13.07770426 },
		{ -0.40510445    , 2.041780255 },
		{ 11.96966384    , 3.342579823 },
		{ -0.615054028   , 10.9235911 },
		{ -1.998813693   , 3.777300472 },
		{ 4.69404902     , -5.58346974 },
		{ 3.864565718    , -11.65520324 },
		{ 8.914591293    , -6.021932174 },
		{ 9.522443396    , 2.083310343 },
		{ 4.936441949    , 8.258857285 },
		{ 5.028926703    , -3.055385439 },
		{ 12.38024444    , 1.380694404 },
		{ 2.686502675    , -8.235556426 }
	});

	testOutputs = MultiSignalVector(vector<vector<float> >
	{
		{ 0  },
		{ 2 },
		{ 2 },
		{ 1 },
		{ 2 },
		{ 2 },
		{ 1 },
		{ 1 },
		{ 0 },
		{ 2 },
		{ 2 },
		{ 0 },
		{ 2 },
		{ 2 },
		{ 1 },
		{ 2 },
		{ 0 },
		{ 2 },
		{ 2 },
		{ 2 },
		{ 2 },
		{ 0 },
		{ 0 },
		{ 2 },
		{ 2 },
		{ 1 },
		{ 0 },
		{ 0 },
		{ 1 },
		{ 0 },
		{ 0 },
		{ 1 },
		{ 0 },
		{ 1 },
		{ 1 },
		{ 1 },
		{ 1 },
		{ 1 },
		{ 2 },
		{ 1 },
		{ 1 },
		{ 0 },
		{ 1 },
		{ 1 },
		{ 1 },
		{ 0 },
		{ 0 },
		{ 0 },
		{ 0 },
		{ 1 },
		{ 2 },
		{ 2 },
		{ 2 },
		{ 0 },
		{ 2 },
		{ 1 },
		{ 1 },
		{ 0 },
		{ 0 },
		{ 1 },
		{ 2 },
		{ 2 },
		{ 1 },
		{ 1 },
		{ 2 },
		{ 0 },
		{ 2 },
		{ 0 },
		{ 2 },
		{ 1 },
		{ 1 },
		{ 2 },
		{ 0 },
		{ 0 },
		{ 1 },
		{ 2 },
		{ 2 },
		{ 0 },
		{ 1 },
		{ 0 },
		{ 1 },
		{ 1 },
		{ 2 },
		{ 0 },
		{ 0 },
		{ 0 },
		{ 0 },
		{ 1 },
		{ 0 },
		{ 2 },
		{ 0 },
		{ 0 },
		{ 1 },
		{ 1 },
		{ 2 },
		{ 2 },
		{ 0 },
		{ 0 },
		{ 2 },
		{ 1 }
	});

	// Normalize the data

	for (size_t i = 0; i < trainingInputs.size(); ++i)
	{
		for (size_t j = 0; j < trainingInputs.signalSize(); ++j)
		{
			trainingInputs[i][j] /= normalScale;
		}
	}

	for (size_t i = 0; i < trainingOutputs.size(); ++i)
	{
		for (size_t j = 0; j < trainingOutputs.signalSize(); ++j)
		{
			trainingOutputs[i][j] += outputOffset;
		}
	}

	for (size_t i = 0; i < testInputs.size(); ++i)
	{
		for (size_t j = 0; j < testInputs.signalSize(); ++j)
		{
			testInputs[i][j] /= normalScale;
		}
	}

	for (size_t i = 0; i < testOutputs.size(); ++i)
	{
		for (size_t j = 0; j < testOutputs.signalSize(); ++j)
		{
			testOutputs[i][j] += outputOffset;
		}
	}
}

