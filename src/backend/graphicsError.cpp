#include "backend/graphicsError.h"


namespace NeuronalNet
{
    bool GraphicsError::m_hadAnError = false;
    void GraphicsError::errorOccured()
    {
        m_hadAnError = true;
    }
    bool GraphicsError::hasError()
    {
        return m_hadAnError;
    }
    void GraphicsError::clearError()
    {
        m_hadAnError = false;
    }
}
