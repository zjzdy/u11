TEMPLATE = app

QT += qml quick widgets

SOURCES += main.cpp \
    crop.cpp

RESOURCES += qml.qrc

# Additional import path used to resolve QML modules in Qt Creator's code model
QML_IMPORT_PATH =

# Default rules for deployment.
include(deployment.pri)

DISTFILES += \
    Chrysanthemum.jpg \
    android/AndroidManifest.xml \
    android/res/values/libs.xml \
    android/build.gradle

HEADERS += \
    crop.h

DEPENDPATH += H:/msys2/soft/x86_build/usr/include
INCLUDEPATH += H:/msys2/soft/x86_build/usr/include
LIBS+=-LG:/msys2/msys32/mingw32/lib -lws2_32 -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -ltiff -ljasper -lpng -ljpeg -lwebp -lz# -ltesseract -llept #-lopencv_imgproc.dll

contains(ANDROID_TARGET_ARCH,armeabi) {
    DEPENDPATH += G:/msys2/msys32/arm/include/
    INCLUDEPATH += G:/msys2/msys32/arm/include/
    LIBS+=-LG:/msys2/msys32/arm/lib/ -LH:/crystax/build/usr/lib \
        G:/msys2/msys32/arm/lib/libopencv_java.so \
        G:/msys2/msys32/arm/lib/libopencv_*.a \#-lopencv_imgproc,-lopencv_highgui,-lopencv_core,-ltesseract,
        -Wl,-Bstatic,-llept,-lIlmImf,-llibtiff,-llibjasper,-llibpng,-llibjpeg,-Bdynamic \
        -lboost_system -lcrystax -lm -llog -ldl
    ANDROID_EXTRA_LIBS = \
        #G:/msys2/msys32/arm/lib/liblzma.so \
        #G:/msys2/msys32/arm/lib/libzim.so \
        #G:/msys2/msys32/arm/lib/libopencv_java.so \
        #G:/msys2/msys32/arm/lib/libopencv_info.so \
        G:/msys2/msys32/arm/lib/libopencv_*.so \
        H:/crystax/build/usr/lib/libcrystax.so \
        H:/crystax/build/usr/lib/libboost_system.so
        #H:/crystax/build/usr/lib/libz.so \
        #G:/msys2/msys32/home/zjzdy/LucenePlusPlus/arm_build/src/core/liblucene++.so
    QMAKE_CXXFLAGS += -DOPENCV_CAMERA_MODULES=off
    QMAKE_CFLAGS += -DOPENCV_CAMERA_MODULES=off
    QMAKE_LFLAGS += -DOPENCV_CAMERA_MODULES=off
    OPENCV_CAMERA_MODULES += off
}

ANDROID_PACKAGE_SOURCE_DIR = $$PWD/android
