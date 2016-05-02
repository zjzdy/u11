#include <QApplication>
#include <QQmlApplicationEngine>
#include <QQmlEngine>
#include <QQuickView>
#include <QQmlContext>
#include "crop.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    QQmlApplicationEngine engine;
    engine.load(QUrl(QStringLiteral("qrc:/main.qml")));
    crop crop1;
    //engine.rootContext()->setContextProperty("crop", &interface);
    engine.rootContext()->setContextProperty("crop", &crop1);

    return app.exec();
}
