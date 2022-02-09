#include "pch.h"
#include "CppUnitTest.h"


using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace NeuronalNetUnitTest
{
	TEST_CLASS(NeuronalNetUnitTest)
	{
	public:
		
		TEST_METHOD(TestMethod1)
		{
			Net net;
			net.build();
		}
	};
}
