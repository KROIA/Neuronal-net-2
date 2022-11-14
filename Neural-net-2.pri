
inc = $$PWD/inc
src = $$PWD/src

CONFIG += object_parallel_to_source

INCLUDEPATH += \
	$$inc \
#	$$inc/backend \
	 
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
    $$inc/backend/netSerializer.h \
	
SOURCES += \
    $$src/backend/backpropNet.cpp \
    $$src/backend/debug.cpp \
    $$src/backend/geneticNet.cpp \
    $$src/backend/multiSignalVector.cpp \
    $$src/backend/net.cpp \
    $$src/backend/neuronIndex.cpp \
    $$src/backend/signalVector.cpp \
    $$src/backend/utilities.cpp \
    $$src/backend/netSerializer.cpp \

# CONFIG(debug, debug|release) {
# #message( "  Copy debug dll\'s: $$SFML_BIN" )
#     QMAKE_PRE_LINK += & copy "\"$$PWD\\x64\\Debug\\Neural-net-2.dll\"" "debug\Neural-net-2.dll"
#     LIBS += $$PWD\\x64\\Debug\\Neural-net-2.dll
# }else{
# #message( "  Copy release dll\'s $$SFML_BIN" )
#     QMAKE_PRE_LINK += & copy "\"$$PWD\\x64\\Release\\Neural-net-2.dll\""  "release\Neural-net-2.dll"
# }

