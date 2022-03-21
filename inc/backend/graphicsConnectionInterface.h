#pragma once

#include "backend/neuronIndex.h"

namespace NeuronalNet
{

	class NET_API GraphicsConnectionInterface
	{
		public:

		virtual void update(float weight, float signal, 
							float minW, float maxW,
							float minS, float maxS) = 0;

		void index(const ConnectionIndex& index)
		{
			m_index = index;
		}
		const ConnectionIndex& index() const
		{
			return m_index;
		}


		protected:
		ConnectionIndex m_index;
	};
}; 