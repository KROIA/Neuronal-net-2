

#include(Extern/SFML.pri)

inc = $$PWD/inc
src = $$PWD/src



INCLUDEPATH += \
	$$inc \
	$$inc/backend \
        #$$inc/graphics
	 
HEADERS += \
    $$inc/neuronalNet.h \
    $$inc/backend/activation.h \
    $$inc/backend/backpropNet.h \
    $$inc/backend/config.h \
    $$inc/backend/debug.h \
    $$inc/backend/geneticNet.h \
    $$inc/backend/graphicsConnectionInterface.h \
    $$inc/backend/graphicsNeuronInterface.h \
    $$inc/backend/multiSignalVector.h \
    $$inc/backend/net.h \
    $$inc/backend/neuronIndex.h \
    $$inc/backend/signalVector.h \
    $$inc/backend/utilities.h \
    #$$inc/graphics/connectionPainter.h \
    #$$inc/graphics/display.h \
    #$$inc/graphics/displayInterface.h \
    #$$inc/graphics/drawable.h \
    #$$inc/graphics/drawableInterface.h \
    #$$inc/graphics/graphicsUtilities.h \
    #$$inc/graphics/netModel.h \
    #$$inc/graphics/neuronPainter.h \
    #$$inc/graphics/pixelPainter.h
	
SOURCES += \
    $$src/backend/backpropNet.cpp \
    $$src/backend/debug.cpp \
    $$src/backend/geneticNet.cpp \
    $$src/backend/multiSignalVector.cpp \
    $$src/backend/net.cpp \
    $$src/backend/neuronIndex.cpp \
    $$src/backend/signalVector.cpp \
    $$src/backend/utilities.cpp \
    #$$src/graphics/connectionPainter.cpp \
    #$$src/graphics/display.cpp \
    #$$src/graphics/drawable.cpp \
    #$$src/graphics/graphicsUtilities.cpp \
    #$$src/graphics/netModel.cpp \
    #$$src/graphics/neuronPainter.cpp \
    #$$src/graphics/pixelPainter.cpp
	
# CONFIG(debug, debug|release) {
# #message( "  Copy debug dll\'s: $$SFML_BIN" )
#     QMAKE_PRE_LINK += & copy "\"$$PWD\\x64\\Debug\\Neural-net-2.dll\"" "debug\Neural-net-2.dll"
#     LIBS += $$PWD\\x64\\Debug\\Neural-net-2.dll
# }else{
# #message( "  Copy release dll\'s $$SFML_BIN" )
#     QMAKE_PRE_LINK += & copy "\"$$PWD\\x64\\Release\\Neural-net-2.dll\""  "release\Neural-net-2.dll"
# }

