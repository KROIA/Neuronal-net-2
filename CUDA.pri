
# #ADD TO PATH: C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64
# # Define output directories
# DESTDIR = cuda
# OBJECTS_DIR = cuda/obj
# CUDA_OBJECTS_DIR = cuda
#
# # Source files
# #SOURCES += src/main.cpp
#
# # This makes the .cu files appear in your project
# OTHER_FILES +=  $$PWD/src/backend/net_kernel.cu
# HEADERS += $$PWD/inc/backend/net_kernel.cuh
# INCLUDEPATH += $$PWD/inc \
#                $$PWD/inc/backend
#
# # CUDA settings <-- may change depending on your system
# CUDA_SOURCES += $$PWD/src/backend/net_kernel.cu
# CUDA_SDK = "C:/ProgramData/NVIDIA Corporation/NVIDIA GPU Computing SDK 4.2/C"   # Path to cuda SDK install
# CUDA_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6"            # Path to cuda toolkit install
# SYSTEM_NAME = x64         # Depending on your system either 'Win32', 'x64', or 'Win64'
# SYSTEM_TYPE = 64            # '32' or '64', depending on your system
# CUDA_ARCH = sm_75           # Type of CUDA architecture, for example 'compute_10', 'compute_11', 'sm_10'
# NVCC_OPTIONS = --use_fast_math -rdc=true
#
# # include paths
# INCLUDEPATH += $$CUDA_DIR/include \
#               ## $$CUDA_SDK/common/inc/ \
#               ## $$CUDA_SDK/../shared/inc/
#
# # library directories
# QMAKE_LIBDIR += $$CUDA_DIR/lib/$$SYSTEM_NAME \
#                # $$CUDA_SDK/common/lib/$$SYSTEM_NAME \
#                # $$CUDA_SDK/../shared/lib/$$SYSTEM_NAME
# # Add the necessary libraries
# LIBS += -lcuda -lcudart -lcudadevrt
#
# # The following library conflicts with something in Cuda
# #QMAKE_LFLAGS_RELEASE = /NODEFAULTLIB:msvcrt.lib
# #QMAKE_LFLAGS_DEBUG   = /NODEFAULTLIB:msvcrtd.lib
#
# # The following makes sure all path names (which often include spaces) are put between quotation marks
# CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')
#
# # Configuration of the Cuda compiler
# CONFIG(debug, debug|release) {
#     # Debug mode
#     cuda_d.input = CUDA_SOURCES
#     cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
#     cuda_d.commands = $$CUDA_DIR/bin/nvcc.exe -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC --machine=$$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
#     cuda_d.dependency_type = TYPE_C
#     QMAKE_EXTRA_COMPILERS += cuda_d
# }
# else {
#     # Release mode
#     cuda.input = CUDA_SOURCES
#     cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
#     cuda.commands = $$CUDA_DIR/bin/nvcc.exe $$NVCC_OPTIONS $$CUDA_INC --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
#     cuda.dependency_type = TYPE_C
#     QMAKE_EXTRA_COMPILERS += cuda
# }

# #----------------------------------------------------------------
# #-------------------------Cuda setup-----------------------------
# #----------------------------------------------------------------
#
# #Enter your gencode here!
# GENCODE = arch=compute_52,code=sm_52
#
# #We must define this as we get some confilcs in minwindef.h and helper_math.h
# DEFINES += NOMINMAX
#
# #set out cuda sources
# CUDA_SOURCES = "$$PWD"/src/backend/net_kernel.cu
#
# #This is to add our .cu files to our file browser in Qt
# SOURCES+=$$PWD/src/backend/net_kernel.cu
# SOURCES-=$$PWD/src/backend/net_kernel.cu
#
# # Path to cuda SDK install
# macx:CUDA_DIR = /Developer/NVIDIA/CUDA-6.5
# linux:CUDA_DIR = /usr/local/cuda-6.5
# win32:CUDA_DIR = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6"
# # Path to cuda toolkit install
# macx:CUDA_SDK = /Developer/NVIDIA/CUDA-6.5/samples
# linux:CUDA_SDK = /usr/local/cuda-6.5/samples
# win32:CUDA_SDK = "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.6"
#
# #Cuda include paths
# INCLUDEPATH += $$CUDA_DIR/include
# #INCLUDEPATH += $$CUDA_DIR/common/inc/
# #INCLUDEPATH += $$CUDA_DIR/../shared/inc/
# #To get some prewritten helper functions from NVIDIA
# win32:INCLUDEPATH += $$CUDA_SDK\common\inc
#
# #cuda libs
# macx:QMAKE_LIBDIR += $$CUDA_DIR/lib
# linux:QMAKE_LIBDIR += $$CUDA_DIR/lib64
# win32:QMAKE_LIBDIR += $$CUDA_DIR\lib\x64
# linux|macx:QMAKE_LIBDIR += $$CUDA_SDK/common/lib
# win32:QMAKE_LIBDIR +=$$CUDA_SDK\common\lib\x64
# LIBS += -lcudart -lcudadevrt
#
# # join the includes in a line
# CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')
#
# # nvcc flags (ptxas option verbose is always useful)
# NVCCFLAGS = --compiler-options  -fno-strict-aliasing --ptxas-options=-v -maxrregcount 20 --use_fast_math
#
# #On windows we must define if we are in debug mode or not
# CONFIG(debug, debug|release) {
# #DEBUG
#     # MSVCRT link option (static or dynamic, it must be the same with your Qt SDK link option)
#     win32:MSVCRT_LINK_FLAG_DEBUG = "/MDd"
#     win32:NVCCFLAGS += -D_DEBUG -Xcompiler $$MSVCRT_LINK_FLAG_DEBUG
# }
# else{
# #Release UNTESTED!!!
#     win32:MSVCRT_LINK_FLAG_RELEASE = "/MD"
#     win32:NVCCFLAGS += -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE
# }
#
# #prepare intermediat cuda compiler
# cudaIntr.input = CUDA_SOURCES
# cudaIntr.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}.o
# #So in windows object files have to be named with the .obj suffix instead of just .o
# #God I hate you windows!!
# win32:cudaIntr.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}.obj
#
# ## Tweak arch according to your hw's compute capability
# cudaIntr.commands = $$CUDA_DIR/bin/nvcc -m64 -g -gencode $$GENCODE -dc $$NVCCFLAGS $$CUDA_INC $$LIBS ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
#
# #Set our variable out. These obj files need to be used to create the link obj file
# #and used in our final gcc compilation
# cudaIntr.variable_out = CUDA_OBJ
# cudaIntr.variable_out += OBJECTS
# cudaIntr.clean = cudaIntrObj/*.o
# win32:cudaIntr.clean = cudaIntrObj/*.obj
#
# QMAKE_EXTRA_UNIX_COMPILERS += cudaIntr
#
# # Prepare the linking compiler step
# cuda.input = CUDA_OBJ
# cuda.output = ${QMAKE_FILE_BASE}_link.o
# win32:cuda.output = ${QMAKE_FILE_BASE}_link.obj
#
# # Tweak arch according to your hw's compute capability
# cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -g -gencode $$GENCODE  -dlink    ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
# cuda.dependency_type = TYPE_C
# cuda.depend_command = $$CUDA_DIR/bin/nvcc -g -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}
# # Tell Qt that we want add more stuff to the Makefile
# QMAKE_EXTRA_UNIX_COMPILERS += cuda


#
#  QT -= gui
#  QT += core
#
#  CONFIG += c++11 console
#  CONFIG -= app_bundle
#  DEFINES += QT_DEPRECATED_WARNINGS
#
#  qnx: target.path = /tmp/$${TARGET}/bin
#  else: unix:!android: target.path = /opt/$${TARGET}/bin
#  !isEmpty(target.path): INSTALLS += target
#
#
#  DESTDIR     = $$PWD/bin
#  OBJECTS_DIR = $$DESTDIR/Obj
#  # C++ flags
#  QMAKE_CXXFLAGS_RELEASE =-03
#
#
#
#  #SOURCES += main.cpp
#
#  CUDA_DIR = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6'
#  INCLUDEPATH  += $$CUDA_DIR/include
#  QMAKE_LIBDIR += $$CUDA_DIR/lib/x64
#  DEPENDPATH   += $$CUDA_DIR/lib
#  LIBS += -lcudart -lcuda
#  CUDA_ARCH = sm_61                # Yeah! I've a new device. Adjust with your compute capability
#
#  CUDA_SOURCES += $$PWD/src/backend/net_kernel.cu
#  HEADERS += $$PWD/inc/backend/net_kernel.cuh
#  INCLUDEPATH += $$PWD/inc/backend
#
#  # Here are some NVCC flags I've always used by default.
#  NVCCFLAGS     = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v
#  CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')
#  cuda.commands = \"$$CUDA_DIR/bin/nvcc\" -m64 -O3 -arch=$$CUDA_ARCH -c $$NVCCFLAGS \
#                  $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
#                  #2>&1
#                  #| sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
#  cuda.dependency_type = TYPE_C
#  cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS ${QMAKE_FILE_NAME}
#   # | sed \"s/^.*: //\"
#  cuda.input = CUDA_SOURCES
#  cuda.output = $${OBJECTS_DIR}/${QMAKE_FILE_BASE}$${QMAKE_EXT_OBJ}
#  QMAKE_EXTRA_COMPILERS += cuda ```
#  message($$cuda.commands)
#  message($$cuda.output)
#  message($$INCLUDEPATH)

# CUDA_BASE= "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6"
# isEmpty( CUDA_BASE ) {
#         error("environment-variable CUDA_PATH is not defined!")
# }
# CUDA_BIN_PATH=$$CUDA_BASE/bin
# CUDA_INC_PATH=$$CUDA_BASE/include
# CUDA_LIB_PATH=$$CUDA_BASE/lib
# #CUDA_LIB_PATH=$$CUDA_BASE/lib64
# !contains(QMAKE_HOST.arch, x86_64) {
#    CUDA_LIB_PATH=$$CUDA_LIB_PATH/win32
# } else {
#    CUDA_LIB_PATH=$$CUDA_LIB_PATH/x64
# }
#
# # GPU architecture
# CUDA_ARCH = sm_35
# CUDA_VARCH = compute_35
#
#
# # Add the necessary libraries
# # CUDA < 9.0
# #CUDA_LIBS=cudart_static nppi nppc
# # CUDA >= 10.0
# CUDA_LIBS=cuda cudart_static nppc curand cudadevrt
#
#
# CUDA_SOURCES += $$PWD/src/backend/net_kernel.cu
# HEADERS += $$PWD/inc/backend/net_kernel.cuh
#
# # setting the CUdaCompiler
# QMAKE_CUC = $$CUDA_BIN_PATH/nvcc.exe
# #QMAKE_CUC = $$CUDA_BIN_PATH/nvcc
# win32 {
#         !exists($$QMAKE_CUC) {
#                 warning("can't find cuda compiler($$QMAKE_CUC)")
#         }
#         !exists($$CUDA_INC_PATH/cuda.h) {
#                 warning("can't find cuda include ($$CUDA_INC_PATH/cuda.h)")
#         }
#         !exists($$CUDA_LIB_PATH/cuda.lib) {
#                 warning("can't find cuda lib ($$CUDA_LIB_PATH/cuda.lib)")
#         }
# }
#
# # Cuda extra-compiler for handling files specified in the CUDA_SOURCES variable
# {
#    cu.name = Cuda Sourcefiles
#         cu.input = CUDA_SOURCES
#         cu.dependency_type = TYPE_C
#         cu.CONFIG += no_link
#         cu.variable_out = OBJECTS
#         isEmpty(QMAKE_CUC) {
#       win32:QMAKE_CUC = $$CUDA_BIN_PATH/nvcc.exe
#       else:QMAKE_CUC = nvcc
#         }
#
#         isEmpty(CU_DIR):CU_DIR = .
#         isEmpty(QMAKE_CPP_MOD_CU):QMAKE_CPP_MOD_CU = cu_
#         isEmpty(QMAKE_EXT_CPP_CU):QMAKE_EXT_CPP_CU = .cu
#         INCLUDEPATH += $$CUDA_INC_PATH
#
#    CONFIG(debug, debug|release) {
#       QMAKE_CUFLAGS += $$QMAKE_CXXFLAGS $$QMAKE_CXXFLAGS_DEBUG $$QMAKE_CXXFLAGS_RTTI_ON $$QMAKE_CXXFLAGS_WARN_ON $$QMAKE_CXXFLAGS_STL_ON
#                 QMAKE_NVVFLAGS += -G
#    }
#    CONFIG(release, debug|release) {
#       QMAKE_CUFLAGS += $$QMAKE_CXXFLAGS $$QMAKE_CXXFLAGS_RELEASE $$QMAKE_CXXFLAGS_RTTI_ON $$QMAKE_CXXFLAGS_WARN_ON $$QMAKE_CXXFLAGS_STL_ON
#    }
#         #since qt5.9 they use /Zc:rvalueCast- /Zc:inline- in the msvc-mkspecs
#         # we have to switch that off!!
#         # else linking __device__ __managed__ vars wont work correctly (in release mode only)
#    QMAKE_CUFLAGS = $$replace(QMAKE_CUFLAGS, -Zc:inline, )
#
#         QMAKE_NVVFLAGS += -gencode arch=$$CUDA_VARCH,code=$$CUDA_ARCH
#         QMAKE_NVVFLAGS += -rdc=true
#         # -keep for holding intermediat files
#         QMAKE_CUEXTRAFLAGS += -Xcompiler $$join(QMAKE_CUFLAGS, ",")
#    !contains(QMAKE_HOST.arch, x86_64) {
#       ## Windows x86 (32bit) specific build here
#       QMAKE_CUEXTRAFLAGS += --machine 32 --debug
#    } else {
#       ## Windows x64 (64bit) specific build here
#       QMAKE_CUEXTRAFLAGS += --machine 64
#    }
#    #QMAKE_CUEXTRAFLAGS += $(DEFINES) $(INCPATH) $$join(QMAKE_COMPILER_DEFINES, " -D", -D)  -Xcompiler #/Zc:__cplusplus   this is for Windows
#    QMAKE_CUEXTRAFLAGS += $(DEFINES) $(INCPATH) -Xcompiler #/Zc:__cplusplus   this is for Windows
#
#         #QMAKE_CUEXTRAFLAGS += -Xcudafe "--diag_suppress=field_without_dll_interface"      this is for Windows
#         #QMAKE_CUEXTRAFLAGS += -Xcudafe "--diag_suppress=code_is_unreachable"         this is for Windows
#         #-Xcudafe "--diag_suppress=boolean_controlling_expr_is_constant"
#
#    CONFIG(debug, debug|release) {
#       CUDA_OBJ_DIR = cuda/debug
#    } else {
#       CUDA_OBJ_DIR = cuda/release
#    }
#    cu.dependency_type = TYPE_C
#         cu.commands = \"$$QMAKE_CUC\" $$QMAKE_NVVFLAGS $$QMAKE_CUEXTRAFLAGS -c -o $${CUDA_OBJ_DIR}/$${QMAKE_CPP_MOD_CU}${QMAKE_FILE_BASE}$${QMAKE_EXT_OBJ} ${QMAKE_FILE_NAME}$$escape_expand(\\n\\t)
#         cu.output = $${# _DIR}/$${QMAKE_CPP_MOD_CU}${QMAKE_FILE_BASE}$${QMAKE_EXT_OBJ}
#         #silent:cu.commands = @echo nvcc ${QMAKE_FILE_IN} && $$cu.commands
#         cu.commands = @echo nvcc ${QMAKE_FILE_IN} && $$cu.commands
#    cu.commands = $$replace(cu.commands,-D__cplusplus=199711L,)
#         QMAKE_EXTRA_COMPILERS += cu
#
#         build_pass|isEmpty(BUILDS):cuclean.depends = compiler_cu_clean
#         else:cuclean.CONFIG += recursive
#         QMAKE_EXTRA_TARGETS += cuclean
#
#    # another compiler-entry for linking the device-code
#         device_link_target.target = $${CUDA_OBJ_DIR}/$${TARGET}_cu_device_link$${QMAKE_EXT_OBJ}
#         #device_link_target.CONFIG += no_check_exist executable
#         for(var, CUDA_SOURCES) {
#           var = $$basename(var)
#           var = $$replace(var,"\\.cu",$${QMAKE_EXT_OBJ})
#           CUDEP += $${CUDA_OBJ_DIR}/$${QMAKE_CPP_MOD_CU}$$var
#         }
#         var = $$basename(CUDEVLINK)
#         var = $$replace(var,"\\.cu",$${QMAKE_EXT_OBJ})
#    CUDEVLINK_OBJ = $${CUDA_OBJ_DIR}/$${QMAKE_CPP_MOD_CU}$$var
#
#         cu_devlink.output = $${CUDA_OBJ_DIR}/$${TARGET}_cu_device_link$${QMAKE_EXT_OBJ}
#         cu_devlink.input = CUDEVLINK
#         cu_devlink.depends = $$CUDEP
#         cu_devlink.dependency_type = TYPE_C
#         cu_devlink.commands = @echo "link cuda device-code" && \"$$QMAKE_CUC\" -dlink -w -gencode arch=$$CUDA_VARCH,code=$$CUDA_ARCH $$QMAKE_CUEXTRAFLAGS -o $$device_link_target.target $$CUDEP
#         cu_devlink.name = cuda_devlink
#         cu_devlink.variable_out = OBJECTS
#         cu_devlink.CONFIG = silent
#         QMAKE_EXTRA_COMPILERS += cu_devlink
#
#    QMAKE_PRE_LINK += $${cu_devlink.commands}
# }
#
# # add the cuda-libraries to the project
# LIBS += -L$$CUDA_LIB_PATH
# for(lnam, CUDA_LIBS) {
#   LIBS+=$$join(lnam, " -l", -l)
# }

# CUDA_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6"
# CUDA_LIBS = -lcudart -lcuda
#
# INCLUDEPATH  += $$CUDA_DIR/include
# QMAKE_LIBDIR += $$CUDA_DIR/lib/x64
# LIBS += $$CUDA_LIBS
#
# CUDA_SOURCES += $$PWD/src/backend/net_kernel.cu
# HEADERS += $$PWD/inc/backend/net_kernel.cuh
#
# #####################################################################
# #                   CUDA compiler configuration                     #
# #####################################################################
#
# # GPU architecture
# SYSTEM_TYPE = 64
# CUDA_ARCH = sm_30
# NVCCOPTIONS = -use_fast_math -O2
#
# # Mandatory flags for stepping through the code
# debug {
#     NVCCOPTIONS += -g -G
# }
#
# # Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
# CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')
#
# cuda.input = CUDA_SOURCES
# cuda.output = ${OBJECTS_DIR}/${QMAKE_FILE_BASE}_cuda.o
# cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCCOPTIONS $$CUDA_INC $$CUDA_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME} 2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
# cuda.dependency_type = TYPE_C
#
# QMAKE_EXTRA_COMPILERS += cuda


