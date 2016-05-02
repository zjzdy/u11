#ifndef CROP_H
#define CROP_H

#include <QObject>
#include <QFile>
#include <QPair>
#include <QImage>
#include <QMap>
#include <QPointF>
#include <QList>
#include <QString>
#include <QStringList>
#include <QVariantList>
#include <QDebug>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <cstdio>

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>


using namespace std;
using namespace cv;
using namespace tesseract;

class crop : public QObject
{
    Q_OBJECT
public:
    explicit crop(QObject *parent = 0);

    Q_INVOKABLE void analyze(QString imagepath, QVariant cropPoints);
    Q_INVOKABLE int radon(QString imagepath);
    Q_INVOKABLE int hough(QString imagepath);
    int otsu(const IplImage *src_image);
    bool getCropPoints(QMap<QString, QPointF> &points, QMap<QString, QVariant> cropPoints, QImage &img);


signals:

public slots:
};

#endif // CROP_H
