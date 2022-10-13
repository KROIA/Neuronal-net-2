#pragma once


#include "backend/neuronIndex.h"

namespace NeuronalNet
{
	
	class NET_API GraphicsNeuronInterface
	{
		public:

		virtual void update(float netinput, float output,
							float minN, float maxN,
							float minO, float maxO) = 0;

		void index(const NeuronIndex& index)
		{
			m_index = index;
		}
		const NeuronIndex& index() const
		{
			return m_index;
		}


		protected:
		NeuronIndex m_index;
	};
};