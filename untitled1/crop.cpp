#include "crop.h"

crop::crop(QObject *parent) : QObject(parent) {}

void crop::analyze(QString imagepath, QVariant cropPoints) {

  imagepath.replace("file:///", "");

  QImage img(imagepath);
  QMap<QString, QPointF> points;

  qreal width = 0;
  qreal height = 0;

  if (getCropPoints(points, cropPoints.toMap(), img)) {
    QLineF topLine(points.value("topLeft"), points.value("topRight"));
    QLineF bottomLine(points.value("bottomLeft"), points.value("bottomRight"));
    QLineF leftLine(points.value("topLeft"), points.value("bottomLeft"));
    QLineF rightLine(points.value("topRight"), points.value("bottomRight"));

    if (topLine.length() > bottomLine.length()) {
      width = topLine.length();
    } else {
      width = bottomLine.length();
    }

    if (topLine.length() > bottomLine.length()) {
      height = leftLine.length();
    } else {
      height = rightLine.length();
    }

    Mat img = imread(imagepath.toStdString());
    int img_height = height;
    int img_width = width;
    qDebug() << img_height << img_width;
    Mat img_trans = Mat::zeros(img_height, img_width, CV_8UC3);

    vector<Point2f> corners(4);
    corners[0] = Point2f(0, 0);
    corners[1] = Point2f(img_width - 1, 0);
    corners[2] = Point2f(0, img_height - 1);
    corners[3] = Point2f(img_width - 1, img_height - 1);
    vector<Point2f> corners_trans(4);
    corners_trans[0] =
        Point2f(points.value("topLeft").x(), points.value("topLeft").y());
    corners_trans[1] =
        Point2f(points.value("topRight").x(), points.value("topRight").y());
    corners_trans[2] =
        Point2f(points.value("bottomLeft").x(), points.value("bottomLeft").y());
    corners_trans[3] = Point2f(points.value("bottomRight").x(),
                               points.value("bottomRight").y());

    // Mat transform = getPerspectiveTransform(corners,corners_trans);
    Mat warpMatrix = getPerspectiveTransform(corners_trans, corners);
    warpPerspective(img, img_trans, warpMatrix, img_trans.size(), INTER_LINEAR,
                    BORDER_CONSTANT); // INTER_CUBIC is better but it is slowly.
    imwrite("trans.jpg", img_trans);
  } else {
    img.save("trans.jpg", "jpg", 100);
  }

  Mat image = cv::imread("trans.jpg", CV_LOAD_IMAGE_GRAYSCALE);
  medianBlur(image, image, 3);
  // 局部二值化

  int blockSize = 25;
  int constValue = 11;
  Mat local;
  adaptiveThreshold(image, local, 255, CV_ADAPTIVE_THRESH_MEAN_C,
                    CV_THRESH_BINARY, blockSize, constValue);

  imwrite("local.jpg", local);
  hough("local.jpg");
  //radon("local.jpg");

  /*
      //灰度均衡化

      IplImage *pSrcImage = cvLoadImage("trans.jpg", CV_LOAD_IMAGE_UNCHANGED);
      IplImage *pGrayImage = cvCreateImage(cvGetSize(pSrcImage), IPL_DEPTH_8U,
  1);
      IplImage *pGrayEqualizeImage = cvCreateImage(cvGetSize(pSrcImage),
  IPL_DEPTH_8U, 1);
      cvCvtColor(pSrcImage, pGrayImage, CV_BGR2GRAY);
      cvEqualizeHist(pGrayImage, pGrayEqualizeImage);
      cvSaveImage("GrayEqualize.jpg",pGrayEqualizeImage);
      IplImage *g_pBinaryImage = cvCreateImage(cvGetSize(pGrayEqualizeImage),
  IPL_DEPTH_8U, 1);
      g_pBinaryImage = cvCreateImage(cvGetSize(pGrayEqualizeImage),
  IPL_DEPTH_8U, 1);
      cvThreshold(pGrayEqualizeImage, g_pBinaryImage, 80, CV_THRESH_BINARY,
  CV_THRESH_BINARY);
      cvSaveImage("g_pBinary.jpg",g_pBinaryImage);
      cvThreshold(pGrayEqualizeImage, g_pBinaryImage, 80, CV_THRESH_BINARY,
  THRESH_OTSU);
      cvSaveImage("g_pBinary1.jpg",g_pBinaryImage);

      Mat image2 = cv::imread("GrayEqualize.jpg", CV_LOAD_IMAGE_GRAYSCALE);

      // 局部二值化
      Mat local2;
      adaptiveThreshold(image2, local2, 255, CV_ADAPTIVE_THRESH_MEAN_C,
  CV_THRESH_BINARY, blockSize, constValue);

      imwrite("local2.jpg", local2);
  /
      QImage img_t("local.jpg");
      QTransform transform;
      img_t = img_t.transformed(transform.rotate(-90));
      img_t.save("rotate.jpg","JPG");
      char *outText;
      tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI();
      // Initialize tesseract-ocr with English, without specifying tessdata path
      if (api->Init(".", "chi_sim")) {
          fprintf(stderr, "Could not initialize tesseract.\n");
          return;
      }
      //api->InitLangMod(".", "chi_sim");

      // Open input image with leptonica library
      IplImage *iplimg =  NULL;
      iplimg = cvLoadImage("rotate.jpg");
      //api->SetPageSegMode(tesseract::PSM_AUTO_OSD);
      api->SetImage((unsigned char*)(iplimg->imageData), iplimg->width,
  iplimg->height,iplimg->nChannels, iplimg->widthStep);
      //Pix *image = pixRead("local.jpg");
      //api->SetImage(image);
      // Get OCR result
      outText = api->GetUTF8Text();
      qDebug()<< outText <<api->MeanTextConf();
      //printf("OCR output:\n%s", outText);

      // Destroy used object and release memory
      api->End();
      delete [] outText;
      //pixDestroy(&image);
  */
}

int crop::hough(QString imagepath)
{
#define GRAY_THRESH 160
#define HOUGH_VOTE 90
    Mat srcImg = imread(imagepath.toStdString(), CV_LOAD_IMAGE_GRAYSCALE);
    if(srcImg.empty())
        return -1;

    Point center(srcImg.cols/2, srcImg.rows/2);
    /*
    Mat rotMatS = getRotationMatrix2D(center, 16, 1.0);
    warpAffine(srcImg, srcImg, rotMatS, srcImg.size(), 1, 0, Scalar(255,255,255));
    imwrite("imageText_R.jpg",srcImg);
    */
    Mat padded;
    int opWidth = getOptimalDFTSize(srcImg.rows);
    int opHeight = getOptimalDFTSize(srcImg.cols);
    copyMakeBorder(srcImg, padded, 0, opWidth-srcImg.rows, 0, opHeight-srcImg.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat comImg;
    //Merge into a double-channel image
    merge(planes,2,comImg);

    //Use the same image as input and output,
    //so that the results can fit in Mat well
    dft(comImg, comImg);

    //Compute the magnitude
    //planes[0]=Re(DFT(I)), planes[1]=Im(DFT(I))
    //magnitude=sqrt(Re^2+Im^2)
    split(comImg, planes);
    magnitude(planes[0], planes[1], planes[0]);

    //Switch to logarithmic scale, for better visual results
    //M2=log(1+M1)
    Mat magMat = planes[0];
    magMat += Scalar::all(1);
    log(magMat, magMat);

    //Crop the spectrum
    //Width and height of magMat should be even, so that they can be divided by 2
    //-2 is 11111110 in binary system, operator & make sure width and height are always even
    magMat = magMat(Rect(0, 0, magMat.cols & -2, magMat.rows & -2));

    //Rearrange the quadrants of Fourier image,
    //so that the origin is at the center of image,
    //and move the high frequency to the corners
    int cx = magMat.cols/2;
    int cy = magMat.rows/2;

    Mat q0(magMat, Rect(0, 0, cx, cy));
    Mat q1(magMat, Rect(0, cy, cx, cy));
    Mat q2(magMat, Rect(cx, cy, cx, cy));
    Mat q3(magMat, Rect(cx, 0, cx, cy));

    Mat tmp;
    q0.copyTo(tmp);
    q2.copyTo(q0);
    tmp.copyTo(q2);

    q1.copyTo(tmp);
    q3.copyTo(q1);
    tmp.copyTo(q3);

    //Normalize the magnitude to [0,1], then to[0,255]
    normalize(magMat, magMat, 0, 1, CV_MINMAX);
    Mat magImg(magMat.size(), CV_8UC1);
    magMat.convertTo(magImg,CV_8UC1,255,0);
    imwrite("imageText_mag.jpg",magImg);

    IplImage IplmagImg(magImg);
    int thresh = otsu(&IplmagImg);
    qDebug()<<thresh;

	/*
    vector<Vec4i> lines; 
    // 检测直线，最小投票为90，线条不短于50，间隙不小于10 
    HoughLinesP(magImg,lines,1,pi180,HOUGH_VOTE,70,10); 
	Scalar color = Scalar(0,255,0);
    // 将检测到的直线在图上画出来 
    vector<Vec4i>::const_iterator it=lines.begin(); 
    Mat linImg(magImg.size(),CV_8UC3);
    while(it!=lines.end()) 
    { 
        Point pt1((*it)[0],(*it)[1]); 
        Point pt2((*it)[2],(*it)[3]); 
        line(linImg,pt1,pt2,color,2); //  线条宽度设置为2 
        ++it; 
    } 
	imwrite("imageText_line.jpg",linImg);

*/
    //Turn into binary image
    threshold(magImg,magImg,thresh+52,255,CV_THRESH_BINARY);
    imwrite("imageText_bin.jpg",magImg);

    //Find lines with Hough Transformation
    vector<Vec2f> lines;
    float pi180 = (float)CV_PI/180;
    Mat linImg(magImg.size(),CV_8UC3);
    HoughLines(magImg,lines,1,pi180,HOUGH_VOTE,0,0);
    int numLines = lines.size();
    for(int l=0; l<numLines; l++)
    {
        float rho = lines[l][0], theta = lines[l][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line(linImg,pt1,pt2,Scalar(255,0,0),3,8,0);


        qDebug() << "the rotation angel1:" << theta << "rho" << rho;
        float angel=theta;
        float pi2 = CV_PI/2;
        if(angel != pi2){
            float angelT = srcImg.rows*tan(angel)/srcImg.cols;
            angel = atan(angelT);
        }
        float angelD = angel*180/(float)CV_PI-90;
        qDebug() << "the rotation angel:" << angelD << "rho" << rho;
    }
    imwrite("imageText_line.jpg",linImg);
    if(lines.size() == 3){
        qDebug() << "found three angels:" << lines[0][1]*180/CV_PI << "\n" << lines[1][1]*180/CV_PI << "\n" << lines[2][1]*180/CV_PI << "\n" ;
    }

    //Find the proper angel from the three found angels
    float angel=0;
    float piThresh = (float)CV_PI/90;
    float pi2 = CV_PI/2;
    for(int l=0; l<numLines; l++)
    {
        float theta = lines[l][1];
        if(abs(theta) < piThresh || abs(theta-pi2) < piThresh)
            continue;
        else{
            angel = theta;
            break;
        }
    }

    //Calculate the rotation angel
    //The image has to be square,
    //so that the rotation angel can be calculate right
    angel = angel<pi2 ? angel : angel-CV_PI;
    if(angel != pi2){
        float angelT = srcImg.rows*tan(angel)/srcImg.cols;
        angel = atan(angelT);
    }
    float angelD = angel*180/(float)CV_PI-90;
    qDebug() << "the rotation angel to be applied:" << "\n" << angelD << "\n";

    //Rotate the image to recover
    Mat rotMat = getRotationMatrix2D(center,angelD,1.0);
    Mat dstImg = Mat::ones(srcImg.size(),CV_8UC3);
    warpAffine(srcImg,dstImg,rotMat,srcImg.size(),1,0,Scalar(255,255,255));
    imwrite("imageText_D.jpg",dstImg);
    return angelD;
}

int crop::otsu(const IplImage *src_image) //大津法求阈值
{
    double sum = 0.0;
    double w0 = 0.0;
    double w1 = 0.0;
    double u0_temp = 0.0;
    double u1_temp = 0.0;
    double u0 = 0.0;
    double u1 = 0.0;
    double delta_temp = 0.0;
    double delta_max = 0.0;

    //src_image灰度级
    int pixel_count[256]={0};
    float pixel_pro[256]={0};
    int threshold = 0;
    uchar* data = (uchar*)src_image->imageData;
    //统计每个灰度级中像素的个数
    for(int i = 0; i < src_image->height; i++)
    {
        for(int j = 0;j < src_image->width;j++)
        {
            pixel_count[(int)data[i * src_image->width + j]]++;
            sum += (int)data[i * src_image->width + j];
        }
    }
    //cout<<"平均灰度："<<sum / ( src_image->height * src_image->width )<<endl;
    //计算每个灰度级的像素数目占整幅图像的比例
    float aaa = 0;
    for(int i = 0; i < 256; i++)
    {
        pixel_pro[i] = (float)pixel_count[i] / ( src_image->height * src_image->width );
        //aaa+=pixel_pro[i];
        //qDebug()<<i<<aaa<<pixel_pro[i] <<pixel_count[i];
    }
    //遍历灰度级[0,255],寻找合适的threshold
    for(int i = 0; i < 256; i++)
    {
        w0 = w1 = u0_temp = u1_temp = u0 = u1 = delta_temp = 0;
        for(int j = 0; j < 256; j++)
        {
            if(j <= i)   //背景部分
            {
                w0 += pixel_pro[j];
                u0_temp += j * pixel_pro[j];
            }
            else   //前景部分
            {
                w1 += pixel_pro[j];
                u1_temp += j * pixel_pro[j];
            }
        }
        u0 = u0_temp / w0;
        u1 = u1_temp / w1;
        delta_temp = (float)(w0 *w1* pow((u0 - u1), 2)) ;
        if(delta_temp > delta_max)
        {
            delta_max = delta_temp;
            threshold = i;
        }
    }
    return threshold;
}

#define PI 3.1425926
int crop::radon(QString imagepath) {
  // Mat  grayMat;
  Mat grayMat = cv::imread(imagepath.toStdString(), CV_LOAD_IMAGE_GRAYSCALE);
  // cvtColor(img,grayMat,COLOR_BGRA2GRAY,4);
  int i, j;
  int nmax, Max;
  int theta, thro;
  int **count;
  int fSinAngle, fCosAngle;
  int Height = (grayMat).rows;
  int Width = (grayMat).cols;
  const int MaxTheta = 20;
  const int MinTheta = -20;
  int dValue[MaxTheta - MinTheta + 1];
  memset(dValue, 0, sizeof(int) * (MaxTheta - MinTheta + 1));
  nmax = (int)sqrt(Height * Height + Width * Width);
  count = new int *[MaxTheta - MinTheta + 1];
  for (theta = 0; theta < MaxTheta - MinTheta + 1; theta++)
  {
    count[theta] = new int[nmax];
    memset(count[theta], 0, sizeof(int) * nmax);
  }
  for (theta = MinTheta; theta < MaxTheta; theta++) {
    fSinAngle = (int)sin(PI * theta / 180);
    fCosAngle = (int)cos(PI * theta / 180);
    for (i = 0; i < Height; i++) {
      for (j = 0; j < Width; j++) {
        if (grayMat.at<uchar>(i, j) == 1) {
          thro = (i * fCosAngle + j * fSinAngle);
          if (thro > 0 && thro < nmax) { // theta+20：目的是使其成为整数
            count[theta + 20][thro]++;
          }
        }
      }
    }
  }
  for (i = 0; i < (MaxTheta - MinTheta + 1); i++)
  {
    for (j = 0; j < nmax; j++)
    {
      dValue[i] += fabs(count[i][j] - count[i][j + 1]);
    }
  }
  Max = 0;//dValue[0];
  for (i = 0; i < (MaxTheta - MinTheta + 1); i++)
  {
    qDebug()<<"n:"<<i-20<<" v:"<<dValue[i];
    if (dValue[i] > Max) {
      Max = dValue[i];
      theta = i - 20;
    }
  }
  qDebug()<<theta<<"Max"<<Max;
  QImage img_t("local.jpg");
  QTransform transform;
  img_t = img_t.transformed(transform.rotate(-theta));
  img_t.save("rotate.jpg","JPG");
  return Max;
}

bool crop::getCropPoints(QMap<QString, QPointF> &points,
                         QMap<QString, QVariant> cropPoints, QImage &img) {

  bool cropNeeded = false;
  // check if the points were moved
  foreach (QVariant corner, cropPoints) {

    QString key = cropPoints.key(corner);
    QPointF point = corner.toPointF();
    points.insert(key, point);

    int w = img.width() - 1;
    int h = img.height() - 1;

    if (key == "topLeft") {
      if (point.x() > 1 || point.y() > 1) {
        cropNeeded = true;
      }
    }

    if (key == "topRight") {
      if (point.x() < w || point.y() > 1) {
        cropNeeded = true;
      }
    }

    if (key == "bottomRight") {
      if (point.x() < w || point.y() < h) {
        cropNeeded = true;
      }
    }

    if (key == "bottomLeft") {
      if (point.x() > 1 || point.y() < h) {
        cropNeeded = true;
      }
    }
  }
  return cropNeeded;
}
