#pragma once
//#include <iostream>
#include <fstream>
#include "backend/geneticNet.h"
#include "backend/backpropNet.h"
namespace NeuronalNet
{
	class NET_API NetSerializer
	{
		public:
		NetSerializer();
		~NetSerializer();

		void setFilePath(const std::string& path);
		const std::string& getFilePath() const;

		bool saveToFile(Net* net);
		bool saveToFile(GeneticNet* net);

		bool readFromFile(Net* net);
		bool readFromFile(GeneticNet* net);


		protected:
		struct NetConfiguration
		{
			size_t inputs;
			size_t hiddenX;
			size_t hiddenY;
			size_t outputs;
			Activation activation;
			Hardware hardware;
			size_t streamSize;

			bool biasEnabled;
		};
		struct BackpropNetConfiguration
		{
			NetConfiguration net;
			float learnParameter;
		};
		struct GeneticNetConfiguration
		{
			NetConfiguration net;
			size_t netCount;
			float mutationChance;
			float mutationFactor;
			float weightBounds;
		};

		struct LoadingData
		{
			bool isGeneticNet;
			bool isBackpropNet;
			bool isNet;
			GeneticNetConfiguration geneticNet;
			BackpropNetConfiguration backpropNet;
			NetConfiguration net;

			std::vector<float> weights;
			std::vector<float> bias;
			std::vector<std::vector<float>> weightsArray;
			std::vector<std::vector<float>> biasArray;

		};

		static LoadingData getFileData(std::vector<std::string> lines);

		static std::string getMetadataString();

		static std::string getNetConfigurationStr(const Net* net);
		static NetConfiguration getNetConfigurationFromString(const std::string& config);

		static std::string getGeneticNetConfigurationStr(const GeneticNet* net);
		static GeneticNetConfiguration getGeneticNetConfigurationFromString(std::string config);
		static std::string getBackpropNetConfigurationStr(const BackpropNet* net);
		static BackpropNetConfiguration getBackpropNetConfigurationFromString(std::string config);

		static std::string getWeigtStr(const Net* net);
		static std::vector<float> getWeightFromString(const std::string& weightStr);

		static std::string getBiasStr(const Net* net);
		static std::vector<float> getBiasFromString(std::string biasStr);

		static std::string getFloatsStr(const float *list, size_t count, char key);
		static std::vector<float> getFloatsFromString(std::string biasStr, char key);
		
		static std::string getKeyValueStringPair(const std::string& value, const std::string& key);
		static std::string extractValueOfKey(std::string str, const std::string& key, bool &success);


		bool writeToFile(const std::vector<std::string>& lines);
		std::vector<std::string> readFromFile(bool &success);

		private:

		std::string m_filePath;
	};
}