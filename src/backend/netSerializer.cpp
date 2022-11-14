#include "backend/netSerializer.h"

namespace NeuronalNet
{

NetSerializer::NetSerializer()
{
	m_filePath = "neuronalNet.net";
}
NetSerializer::~NetSerializer()
{

}

void NetSerializer::setFilePath(const std::string& path)
{
	m_filePath = path;
}
const std::string& NetSerializer::getFilePath() const
{
	return m_filePath;
}

bool NetSerializer::saveToFile(Net* net)
{
	if (!net)
	{
		PRINT_ERROR("Can't save net, because it is nullptr");
		return false;
	}
	std::string metadata = getMetadataString();
	std::string configData;
	BackpropNet *backPropNet = dynamic_cast<BackpropNet*>(net);
	if (backPropNet)
	{
		configData = getBackpropNetConfigurationStr(backPropNet);
	}
	else
	{
		configData = getNetConfigurationStr(net);
	}
	std::string weights = getWeigtStr(net);
	std::string bias = getBiasStr(net);

	return writeToFile({ metadata,configData,weights,bias });
}

bool NetSerializer::readFromFile(Net* net)
{
	if (!net)
	{
		PRINT_ERROR("Can't load into a nullptr net");
		return false;
	}
	BackpropNet *backpropNet = dynamic_cast<BackpropNet*>(net);
	bool loadSuccess;
	std::vector<std::string> lines = readFromFile(loadSuccess);
	if (!loadSuccess)
		return false;

	LoadingData data = getFileData(lines);

	if (data.isBackpropNet && backpropNet == nullptr)
	{
		PRINT_ERROR("The file: \"" << m_filePath << "\" is for a BackpropNet but the given Net* is not a BackpropNet*");
		return false;
	}
	if (!data.isBackpropNet && backpropNet)
	{
		PRINT_ERROR("The file: \"" << m_filePath << "\" is not for a BackpropNet but the given Net* is a BackpropNet*");
		return false;
	}
	if (data.isGeneticNet)
	{
		PRINT_ERROR("The file: \"" << m_filePath << "\" is for a GeneticNet, call bool readFromFile(GeneticNet* net) instead");
		return false;
	}

	NetConfiguration config;
	if (data.isBackpropNet && backpropNet)
	{
		config = data.backpropNet.net;
		backpropNet->setLearnParameter(data.backpropNet.learnParameter);
	}

	net->setDimensions(config.inputs, config.hiddenX, config.hiddenY, config.outputs);
	net->setActivation(config.activation);
	net->setHardware(config.hardware);
	net->setStreamSize(config.streamSize);
	net->enableBias(config.biasEnabled);
	bool successBuild = net->build();
	if (successBuild)
	{
		net->setWeight(data.weights);
		net->setBias(data.bias);
	}
	return successBuild;
}
bool NetSerializer::readFromFile(GeneticNet* net)
{
	if (!net)
	{
		PRINT_ERROR("Can't load into a nullptr net");
		return false;
	}
	bool loadSuccess;
	std::vector<std::string> lines = readFromFile(loadSuccess);
	if (!loadSuccess)
		return false;

	LoadingData data = getFileData(lines);

	if (data.isBackpropNet)
	{
		PRINT_ERROR("The file: \"" << m_filePath << "\" is for a BackpropNet but the given Net* a GeneticNet*");
		return false;
	}
	if (data.isNet)
	{
		PRINT_ERROR("The file: \"" << m_filePath << "\" is for a Net but the given Net* a GeneticNet*");
		return false;
	}
	if (!data.isGeneticNet)
	{
		PRINT_ERROR("The file: \"" << m_filePath << "\" is not for a GeneticNet");
		return false;
	}
	NetConfiguration config = data.geneticNet.net;
	if (data.geneticNet.netCount > data.weightsArray.size())
	{
		PRINT_ERROR("The file: \"" << m_filePath << "\" has missing weight data. Weights for "
					<< data.geneticNet.netCount << " nets needed and only "
					<< data.weightsArray.size() << " provided");
		return false;
	}
	if (data.geneticNet.netCount > data.biasArray.size())
	{
		PRINT_ERROR("The file: \"" << m_filePath << "\" has missing bias data. Bias values for "
					<< data.geneticNet.netCount << " nets needed and only "
					<< data.biasArray.size() << " provided");
		return false;
	}

	net->setNetCount(data.geneticNet.netCount);

	net->setDimensions(config.inputs, config.hiddenX, config.hiddenY, config.outputs);
	net->setActivation(config.activation);
	net->setHardware(config.hardware);
	net->setStreamSize(config.streamSize);
	net->enableBias(config.biasEnabled);
	
	net->setMutationChance(data.geneticNet.mutationChance);
	net->setMutationFactor(data.geneticNet.mutationFactor);
	net->setWeightBounds(data.geneticNet.weightBounds);
	

	bool successBuild = net->build();
	if (successBuild)
	{
		for (size_t i = 0; i < data.geneticNet.netCount; ++i)
		{
			net->setWeight(i,data.weightsArray[i]);
			net->setBias(i,data.biasArray[i]);
		}
	}
	return successBuild;
}
NetSerializer::LoadingData NetSerializer::getFileData(std::vector<std::string> lines)
{
	LoadingData data;

	std::string configStr;
	int brackedsOpenCount = 0;
	int bracketsCloseCount = 0;
	size_t firstBracketLine = std::string::npos;
	size_t lastBracketLine = std::string::npos;

	std::vector<float> weights;
	std::vector<float> bias;

	bool readConfig = false;
	for (size_t i = 0; i < lines.size(); ++i)
	{
		if (lines[i].find("//") != std::string::npos)
		{
			lines[i] = lines[i].substr(0, lines[i].find("//"));
		}
		if (lines[i].find("BackpropNet") != std::string::npos)
			data.isBackpropNet = true;
		if (lines[i].find("GeneticNet") != std::string::npos)
			data.isGeneticNet = true;
		
		for (size_t j = 0; j < lines[i].size(); ++j)
		{
			if (lines[i][j] == '{')
			{
				if (firstBracketLine == std::string::npos)
					firstBracketLine = i;
				brackedsOpenCount++;
				readConfig = true;
			}
			else if (lines[i][j] == '}')
			{
				lastBracketLine = i;
				bracketsCloseCount++;

			}
		}
		if (readConfig)
		{
			configStr += lines[i];
		}
		else
		{
			if (data.isGeneticNet)
			{
				if (lines[i].find("W") != std::string::npos)
				{
					data.weightsArray.push_back(getWeightFromString(lines[i]));
				}
				else if (lines[i].find("B") != std::string::npos)
				{
					data.biasArray.push_back(getBiasFromString(lines[i]));
				}
			}
			else
			{
				if (lines[i].find("W") != std::string::npos)
				{
					data.weights = getWeightFromString(lines[i]);
				}
				else if (lines[i].find("B") != std::string::npos)
				{
					data.bias = getBiasFromString(lines[i]);
				}
			}
		}
		if (bracketsCloseCount == brackedsOpenCount)
			readConfig = false;

	}
	if (!data.isGeneticNet && !data.isBackpropNet)
	{
		data.isNet = true;
		data.net = getNetConfigurationFromString(configStr);
	}
	else if (data.isBackpropNet)
	{
		data.backpropNet = getBackpropNetConfigurationFromString(configStr);
	}
	else if (data.isGeneticNet)
	{
		data.geneticNet = getGeneticNetConfigurationFromString(configStr);
	}
	return data;
}

std::string NetSerializer::getMetadataString()
{
	return "// Neural-net version: " + Net::getVersion();
}
std::string NetSerializer::getNetConfigurationStr(const Net* net)
{
	if (!net)
		return "";
	std::string str = "Net\n{\n";
	str += getKeyValueStringPair(std::to_string(net->getInputCount()), "inputs") + "\n";
	str += getKeyValueStringPair(std::to_string(net->getHiddenXCount()), "hiddenX") + "\n";
	str += getKeyValueStringPair(std::to_string(net->getHiddenYCount()), "hiddenY") + "\n";
	str += getKeyValueStringPair(std::to_string(net->getOutputCount()), "outputs") + "\n";
	str += getKeyValueStringPair(std::to_string((int)net->getActivation()), "activation") + "\n";
	str += getKeyValueStringPair(std::to_string((int)net->getHardware()), "hardware") + "\n";
	str += getKeyValueStringPair(std::to_string(net->getStreamSize()), "streamSize") + "\n";
	str += getKeyValueStringPair(std::to_string((int)net->isBiasEnabled()), "biasEnabled") + "\n";
	str += "}\n";
}
NetSerializer::NetConfiguration NetSerializer::getNetConfigurationFromString(const std::string& config)
{
	NetConfiguration conf;
	conf.inputs = 1;
	conf.hiddenX = 0;
	conf.hiddenY = 0;
	conf.outputs = 1;
	conf.activation = Activation::linear;
	conf.hardware = Hardware::cpu;
	conf.streamSize = 1;
	conf.biasEnabled = true;

	if (config.find("Net") == std::string::npos)
	{
		PRINT_ERROR("Can't find configuration data with key: \"Net\" in the string");
		return conf;
	}

	bool success = true;

	std::string inputsStr = extractValueOfKey(config, "inputs", success);
	long long inputs = 2;
	if (success) inputs = std::atoll(inputsStr.c_str());
	if (inputs < 1)
	{
		PRINT_ERROR("Out of range, parameter \"inputs\" = " << inputs << " minimum is 1");
		inputs = 1;
	}
	conf.inputs = inputs;

	std::string hiddenXStr = extractValueOfKey(config, "hiddenX", success);
	long long hiddenX = 0;
	if (success) hiddenX = std::atoll(hiddenXStr.c_str());
	if (hiddenX < 0)
	{
		PRINT_ERROR("Out of range, parameter \"hiddenX\" = " << hiddenX << " minimum is 0");
		hiddenX = 0;
	}
	conf.hiddenX = hiddenX;

	std::string hiddenYStr = extractValueOfKey(config, "hiddenY", success);
	long long hiddenY = 0;
	if (success) hiddenY = std::atoll(hiddenYStr.c_str());
	if (hiddenY < 0)
	{
		PRINT_ERROR("Out of range, parameter \"hiddenY\" = " << hiddenY << " minimum is 0");
		hiddenY = 0;
	}
	conf.hiddenY = hiddenY;

	std::string outputsStr = extractValueOfKey(config, "outputs", success);
	long long outputs = 0;
	if (success) outputs = std::atoll(outputsStr.c_str());
	if (outputs < 1)
	{
		PRINT_ERROR("Out of range, parameter \"outputs\" = " << outputs << " minimum is 1");
		outputs = 1;
	}
	conf.outputs = outputs;

	std::string streamsStr = extractValueOfKey(config, "streamSize", success);
	long long streamSize = 0;
	if (success) streamSize = std::atoll(streamsStr.c_str());
	if (streamSize < 1)
	{
		PRINT_ERROR("Out of range, parameter \"streamSize\" = " << streamSize << " minimum is 1");
		streamSize = 1;
	}
	conf.streamSize = streamSize;

	std::string activationStr = extractValueOfKey(config, "activation", success);
	long long activation = 0;
	if (success) activation = std::atoll(activationStr.c_str());
	if (activation < 0)
	{
		PRINT_ERROR("Out of range, parameter \"activation\" = " << activation << " minimum is 0");
		activation = 0;
	}
	if (activation >= (int)Activation::count)
	{
		PRINT_ERROR("Out of range, parameter \"activation\" = " << activation << " maximum is "<< (int)Activation::count-1);
		activation = 0;
	}
	conf.activation = (Activation)activation;

	std::string hardwareStr = extractValueOfKey(config, "hardware", success);
	long long hardware = 0;
	if (success) hardware = std::atoll(hardwareStr.c_str());
	if (hardware < 0)
	{
		PRINT_ERROR("Out of range, parameter \"hardware\" = " << hardware << " minimum is 0");
		hardware = 0;
	}
	if (hardware >= (int)Hardware::count)
	{
		PRINT_ERROR("Out of range, parameter \"hardware\" = " << hardware << " maximum is " << (int)Hardware::count - 1);
		hardware = 0;
	}
	conf.hardware = (Hardware)hardware;


	std::string biasStr = extractValueOfKey(config, "biasEnabled", success);
	long long bias = 0;
	if (success) bias = std::atoll(biasStr.c_str());
	if (bias)
		conf.biasEnabled = true;
	return conf;
}

std::string NetSerializer::getGeneticNetConfigurationStr(const GeneticNet* net)
{
	if (!net)
		return "";
	std::string str = "GeneticNet\n{\n";
	str += getNetConfigurationStr(net->getNet(0)) + "\n";
	str += getKeyValueStringPair(std::to_string(net->getNetCount()), "netCount") + "\n";
	str += getKeyValueStringPair(std::to_string(net->getMutatuionChance()), "mutationChance") + "\n";
	str += getKeyValueStringPair(std::to_string(net->getMutationFactor()), "mutationFactor") + "\n";
	str += getKeyValueStringPair(std::to_string(net->getWeightBounds()), "weightBounds") + "\n";
	str += "}\n";
	return str;
}
NetSerializer::GeneticNetConfiguration NetSerializer::getGeneticNetConfigurationFromString(std::string config)
{
	GeneticNetConfiguration conf;
	conf.netCount = 2;
	conf.mutationChance = 0.1;
	conf.mutationFactor = 0.01;
	conf.weightBounds = 4;
	std::string titleKey = "GeneticNet{";
	if (config.find(titleKey) == std::string::npos)
	{
		PRINT_ERROR("Can't find configuration data with key: \"GeneticNet\" in the string");
		return conf;
	}
	config = config.substr(config.find(titleKey) + titleKey.size());

	conf.net = getNetConfigurationFromString(config);
	bool success = true;
	std::string netCountStr = extractValueOfKey(config, "netCount", success);
	long long netCount = 2;
	if (success) netCount = std::atol(netCountStr.c_str());
	if (netCount < 2)
	{
		PRINT_ERROR("Out of range, parameter \"netCount\" = "<< netCount << " minimum is 2");
		netCount = 2;
	}
	conf.netCount = netCount;

	std::string mutationChanceStr = extractValueOfKey(config, "mutationChance", success);
	if (success) conf.mutationChance = std::stof(mutationChanceStr);

	std::string mutationFactorStr = extractValueOfKey(config, "mutationFactor", success);
	if (success) conf.mutationFactor = std::stof(mutationFactorStr);

	std::string weightBoundsStr = extractValueOfKey(config, "weightBounds", success);
	if (success) conf.weightBounds = std::stof(weightBoundsStr);
	return conf;
}
std::string NetSerializer::getBackpropNetConfigurationStr(const BackpropNet* net)
{
	if (!net)
		return "";
	std::string str = "BackpropNet\n{\n";
	str += getNetConfigurationStr(net) + "\n";
	str += getKeyValueStringPair(std::to_string(net->getLearnParameter()), "learnParameter") + "\n";
	str += "}\n";
	return str;
}
NetSerializer::BackpropNetConfiguration NetSerializer::getBackpropNetConfigurationFromString(std::string config)
{
	BackpropNetConfiguration conf;
	conf.learnParameter = 0.01;
	std::string titleKey = "BackpropNet{";
	if (config.find(titleKey) == std::string::npos)
	{
		PRINT_ERROR("Can't find configuration data with key: \"BackpropNet\" in the string");
		return conf;
	}
	
	config = config.substr(config.find(titleKey) + titleKey.size());

	conf.net = getNetConfigurationFromString(config);
	bool success = true;
	std::string learnParamStr = extractValueOfKey(config, "learnParameter", success);
	if (success)
		conf.learnParameter = std::stof(learnParamStr);
	return conf;
}

std::string NetSerializer::getWeigtStr(const Net* net)
{
	if (!net)
		return "";
	return getFloatsStr(net->getWeight(), net->getWeightSize(), 'W');
}
std::vector<float> NetSerializer::getWeightFromString(const std::string& weightStr)
{
	return getFloatsFromString(weightStr, 'W');
}

std::string NetSerializer::getBiasStr(const Net* net)
{
	if (!net)
		return "";
	return getFloatsStr(net->getBias(), net->getNeuronCount(), 'B');
}
std::vector<float> NetSerializer::getBiasFromString(std::string biasStr)
{
	return getFloatsFromString(biasStr, 'B');
}

std::string NetSerializer::getFloatsStr(const float* list, size_t count, char key)
{
	std::string str;
	if (!list)
		return str;
	str = key + std::to_string(count) + ":";
	str.reserve(count * 16);
	for (size_t i = 0; i < count; ++i)
	{
		str += std::to_string(list[i]) + ", ";
	}
	return str;
}
std::vector<float> NetSerializer::getFloatsFromString(std::string biasStr, char key)
{
	std::vector<float> values;

	if (biasStr.find(key) == std::string::npos)
	{
		PRINT_ERROR("Can't find data with key: \""<<key<<"\" in the string");
		return values;
	}
	size_t bKey = biasStr.find(key);
	size_t endKey = biasStr.find(":");
	std::string countStr = biasStr.substr(bKey, endKey - bKey);
	long long count = std::atoll(countStr.c_str());
	if (count < 0)
	{
		PRINT_ERROR("Value amount can't be negative");
		return values;
	}
	biasStr = biasStr.substr(endKey + 1);
	values.reserve(count);
	for (size_t i = 0; i < count; ++i)
	{
		size_t commaIndex = biasStr.find(",");
		if (commaIndex == std::string::npos)
		{
			PRINT_ERROR("Can't load all values with key: \"" << key << "\", key \",\" is missing after " << i + 1 << " bias values");
			return std::vector<float>();
		}
		std::string valueStr = biasStr.substr(0, commaIndex);
		biasStr = biasStr.substr(commaIndex + 1);
		float value = std::stof(valueStr);
		values.push_back(value);
	}
	return values;
}
std::string NetSerializer::getKeyValueStringPair(const std::string& value, const std::string& key)
{
	return key + " = " + value +";";
}
std::string NetSerializer::extractValueOfKey(std::string str, const std::string& key, bool& success)
{
	if (str.find(key) == std::string::npos)
	{
		PRINT_ERROR("Can't find key: \""<<key<<"\" in the string: \""<<str<<"\"");
		success = false;
		return "";
	}
	str = str.substr(str.find(key));
	size_t beginValue = str.find("=");
	if (beginValue == std::string::npos)
	{
		PRINT_ERROR("Can't find \"=\" after the key: \""<<key<<"\" in the string : \"" << str << "\"");
		success = false;
		return "";
	}
	size_t endValue = str.find(";");
	if (beginValue == std::string::npos)
	{
		PRINT_ERROR("Can't find \";\" after the key: \"" << key << "\" in the string : \"" << str << "\"");
		success = false;
		return "";
	}
	
	std::string valueStr = str.substr(beginValue + 1, endValue - beginValue - 1);
	while (valueStr[0] == ' ')
	{
		valueStr = valueStr.substr(1);
	}
	success = true;
	return valueStr;
}

bool NetSerializer::writeToFile(const std::vector<std::string>& lines)
{
	std::ofstream file(m_filePath.c_str(), std::ios::out);
	if (!file.is_open())
	{
		PRINT_ERROR("Can't write to file: \"" + m_filePath + "\"");
		return false;
	}
	for (size_t i = 0; i < lines.size(); ++i)
	{
		file << lines[i] << "\n";
	}
	file.close();
	return true;
}
std::vector<std::string> NetSerializer::readFromFile(bool& success)
{
	std::ifstream  file(m_filePath.c_str(), std::ios::in);
	std::vector<std::string> lines;
	if (!file.is_open())
	{
		PRINT_ERROR("Can't read from file: \"" + m_filePath + "\"");
		success = false;
		return lines;
	}
	std::string line;
	while (std::getline(file, line))
	{
		lines.push_back(line);
	}
	file.close();
	success = true;
	return lines;
}


}