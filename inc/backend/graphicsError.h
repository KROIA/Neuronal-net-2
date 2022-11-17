#pragma once

namespace NeuronalNet
{
    class GraphicsError
    {
        public:
            static void errorOccured();
            static bool hasError();
            static void clearError();

        private:
            static bool m_hadAnError;
    };

}
