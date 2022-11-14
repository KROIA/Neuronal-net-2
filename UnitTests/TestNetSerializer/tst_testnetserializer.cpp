#include <QtTest>

// add necessary includes here
#include "neuronalNet.h"
using namespace NeuronalNet;

class TestNetSerializer : public QObject
{
        Q_OBJECT

    public:
        TestNetSerializer();
        ~TestNetSerializer();

    private slots:
        void saveLoadNet();
        void saveLoadBackpropNet();
        void saveLoadGeneticNet();

    private:
        void compareWeights(Net *toTest, Net* original, bool &pass);
        void compareBias(Net *toTest, Net* original, bool &pass);
        bool equalFloat(float a, float b, float tollerance = 0.00001);

};

TestNetSerializer::TestNetSerializer()
{

}

TestNetSerializer::~TestNetSerializer()
{

}

void TestNetSerializer::saveLoadNet()
{

    for(int i=0; i<10; ++i)
    {
        Net saveNet;
        size_t inputs = rand()%10 + 1;
        size_t hiddenX = rand()%10;
        size_t hiddenY = rand()%10;
        size_t outputs = rand()%10 +1;
        size_t streamSize = rand() %10 +1;
        bool biasEnabled = rand() % 2;
        Activation act = (Activation)(rand() % (int)Activation::count);
        Hardware hardware = (Hardware)(rand() % (int)Hardware::count);

        saveNet.setDimensions(inputs, hiddenX, hiddenY, outputs);
        saveNet.setStreamSize(streamSize);
        saveNet.setActivation(act);
        saveNet.setHardware(hardware);
        saveNet.enableBias(biasEnabled);
        QVERIFY(saveNet.build());

        // Save net data
        NetSerializer saveSerializer;
        saveSerializer.setFilePath("netSave1.net");
        QVERIFY(saveSerializer.saveToFile(&saveNet));

        // load net data
        Net loadNet;
        NetSerializer loadSerializer;
        loadSerializer.setFilePath(saveSerializer.getFilePath());

        QVERIFY(saveSerializer.readFromFile(&loadNet));

        if(hiddenX == 0 || hiddenY == 0)
        {
            hiddenX = 0;
            hiddenY = 0;
        }

        // check net data
        QCOMPARE(loadNet.getInputCount(), inputs);
        QCOMPARE(loadNet.getHiddenXCount(), hiddenX);
        QCOMPARE(loadNet.getHiddenYCount(), hiddenY);
        QCOMPARE(loadNet.getOutputCount(), outputs);
        QCOMPARE(loadNet.getStreamSize(), streamSize);
        QCOMPARE(loadNet.getActivation(), act);
      //  QCOMPARE(loadNet.getHardware(), hardware);
        QCOMPARE(loadNet.isBiasEnabled(), biasEnabled);
        bool pass = false;
        compareWeights(&loadNet, &saveNet, pass);
        QVERIFY(pass);
        compareBias(&loadNet, &saveNet, pass);
        QVERIFY(pass);
    }

}
void TestNetSerializer::saveLoadBackpropNet()
{
    for(int i=0; i<10; ++i)
    {
        BackpropNet saveNet;
        size_t inputs = rand()%10 + 1;
        size_t hiddenX = rand()%10;
        size_t hiddenY = rand()%10;
        size_t outputs = rand()%10 +1;
        size_t streamSize = rand() %10 +1;
        bool biasEnabled = rand() % 2;
        Activation act = (Activation)(rand() % (int)Activation::count);
        Hardware hardware = (Hardware)(rand() % (int)Hardware::count);
        float learnRate = Net::getRandomValue(0.001,1);

        saveNet.setDimensions(inputs, hiddenX, hiddenY, outputs);
        saveNet.setStreamSize(streamSize);
        saveNet.setActivation(act);
        saveNet.setHardware(hardware);
        saveNet.enableBias(biasEnabled);
        saveNet.setLearnParameter(learnRate);
        QVERIFY(saveNet.build());

        // Save net data
        NetSerializer saveSerializer;
        saveSerializer.setFilePath("netSave2.net");
        QVERIFY(saveSerializer.saveToFile(&saveNet));

        // load net data
        BackpropNet loadNet;
        NetSerializer loadSerializer;
        loadSerializer.setFilePath(saveSerializer.getFilePath());

        QVERIFY(saveSerializer.readFromFile(&loadNet));

        if(hiddenX == 0 || hiddenY == 0)
        {
            hiddenX = 0;
            hiddenY = 0;
        }

        // check net data
        QCOMPARE(loadNet.getInputCount(), inputs);
        QCOMPARE(loadNet.getHiddenXCount(), hiddenX);
        QCOMPARE(loadNet.getHiddenYCount(), hiddenY);
        QCOMPARE(loadNet.getOutputCount(), outputs);
        QCOMPARE(loadNet.getStreamSize(), streamSize);
        QCOMPARE(loadNet.getActivation(), act);
        //QCOMPARE(loadNet.getHardware(), hardware);
        QCOMPARE(loadNet.isBiasEnabled(), biasEnabled);
        QVERIFY(equalFloat(loadNet.getLearnParameter(), learnRate));
        bool pass = false;
        compareWeights(&loadNet, &saveNet, pass);
        QVERIFY(pass);
        compareBias(&loadNet, &saveNet, pass);
        QVERIFY(pass);
    }
}
void TestNetSerializer::saveLoadGeneticNet()
{
    for(int i=0; i<10; ++i)
    {
        size_t netCount = rand() %20 + 2;
        GeneticNet saveNet(netCount);
        size_t inputs = rand()%10 + 1;
        size_t hiddenX = rand()%10;
        size_t hiddenY = rand()%10;
        size_t outputs = rand()%10 +1;
        size_t streamSize = rand() %10 +1;
        bool biasEnabled = rand() % 2;
        Activation act = (Activation)(rand() % (int)Activation::count);
        Hardware hardware = (Hardware)(rand() % (int)Hardware::count);
        float mutationChance = Net::getRandomValue(0.001,1);
        float mutationFactor = Net::getRandomValue(0.001,1);
        float weightBounds = Net::getRandomValue(2,5);

        saveNet.setDimensions(inputs, hiddenX, hiddenY, outputs);
        saveNet.setStreamSize(streamSize);
        saveNet.setActivation(act);
        saveNet.setHardware(hardware);
        saveNet.enableBias(biasEnabled);
        saveNet.setMutationChance(mutationChance);
        saveNet.setMutationFactor(mutationFactor);
        saveNet.setWeightBounds(weightBounds);
        QVERIFY(saveNet.build());

        // Save net data
        NetSerializer saveSerializer;
        saveSerializer.setFilePath("netSave3.net");
        QVERIFY(saveSerializer.saveToFile(&saveNet));

        // load net data
        GeneticNet loadNet(2);
        NetSerializer loadSerializer;
        loadSerializer.setFilePath(saveSerializer.getFilePath());

        QVERIFY(saveSerializer.readFromFile(&loadNet));

        if(hiddenX == 0 || hiddenY == 0)
        {
            hiddenX = 0;
            hiddenY = 0;
        }

        // check net data
        QCOMPARE(loadNet.getInputCount(), inputs);
        QCOMPARE(loadNet.getHiddenXCount(), hiddenX);
        QCOMPARE(loadNet.getHiddenYCount(), hiddenY);
        QCOMPARE(loadNet.getOutputCount(), outputs);
        QCOMPARE(loadNet.getStreamSize(), streamSize);
        QCOMPARE(loadNet.getActivation(), act);
        //QCOMPARE(loadNet.getHardware(), hardware);
        QCOMPARE(loadNet.isBiasEnabled(), biasEnabled);
        QVERIFY(equalFloat(loadNet.getMutatuionChance(), mutationChance));
        QVERIFY(equalFloat(loadNet.getMutationFactor(), mutationFactor));
        QVERIFY(equalFloat(loadNet.getWeightBounds(), weightBounds));
        QCOMPARE(loadNet.getNetCount(), netCount);
        bool pass = false;
        for(size_t i=0; i<netCount; ++i)
        {
            compareWeights(loadNet.getNet(i), saveNet.getNet(i), pass);
            QVERIFY(pass);
            compareBias(loadNet.getNet(i), saveNet.getNet(i), pass);
            QVERIFY(pass);
        }

    }
}

void TestNetSerializer::compareWeights(Net *toTest, Net* original, bool &pass)
{
    size_t weightCount1 = toTest->getWeightSize();
    size_t weightCount2 = original->getWeightSize();
    pass = true;
    if(weightCount1 != weightCount2)
    {
        pass = false;
        QFAIL((QString("Weightcount not equal: originalSize: ")+QString::number(weightCount2) + QString(" toTest: ")+QString::number(weightCount1)).toStdString().c_str());
    }
    const float *w1 = toTest->getWeight();
    const float *w2 = original->getWeight();
    for(size_t i=0; i<weightCount1; ++i)
    {
        if(!equalFloat(w1[i], w2[i]))
        {
            pass = false;
            QFAIL((QString("original weight[")+QString::number(i) + QString("] = ")+QString::number(w1[i])+QString(" != ")+QString::number(w2[i])).toStdString().c_str());
        }
    }
}
void TestNetSerializer::compareBias(Net *toTest, Net* original, bool &pass)
{
    size_t biasCount1 = toTest->getNeuronCount();
    size_t biasCount2 = original->getNeuronCount();
    pass = true;
    if(biasCount1 != biasCount2)
    {
        pass = false;
        QFAIL((QString("Neuroncount not equal: originalSize: ")+QString::number(biasCount2) + QString(" toTest: ")+QString::number(biasCount1)).toStdString().c_str());
    }
    const float *b1 = toTest->getBias();
    const float *b2 = original->getBias();
    for(size_t i=0; i<biasCount1; ++i)
    {
        if(!equalFloat(b1[i], b2[i]))
        {
            pass = false;
            QFAIL((QString("original bias[")+QString::number(i) + QString("] = ")+QString::number(b1[i])+QString(" != ")+QString::number(b2[i])).toStdString().c_str());
        }
    }
}
bool TestNetSerializer::equalFloat(float a, float b, float tollerance)
{
    if(a == b)
        return true;
    if(abs(a-b)<tollerance)
        return true;
    return false;
}

QTEST_APPLESS_MAIN(TestNetSerializer)

#include "tst_testnetserializer.moc"
