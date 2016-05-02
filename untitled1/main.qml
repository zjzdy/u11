import QtQuick 2.0
import QtQuick.Controls 1.2
import QtQuick.Dialogs 1.2
import QtQuick.Window 2.2

Window {
    height: 500
    width: 500
    visible: true
    property var cropPoints: {"topLeft": Qt.point(0, 0)};
    property string curPoint: "";
    FileDialog {
        id: file
        selectExisting: true
        selectMultiple: false
        selectFolder: false
        title: "Please choose a image"
        nameFilters: [ "Image files (*.jpg *.png)", "All files (*)" ]
        onAccepted: {
            cropView.source = file.fileUrl
        }
    }

Item
{
    anchors.fill: parent
    Item {

        id: zoomArea
        width: 110
        height: 110
        anchors.fill: mask
        clip: true

        Rectangle {
            id: mask
            anchors.top: parent.top
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.leftMargin: 540 / 2 - width / 2
            anchors.topMargin: -zoomArea.height
            width: zoomArea.width
            height: zoomArea.height
            border.width: 1
            clip: true
        }

        Image {
            id: zoom
            cache: false
            width: 5 * cropView.paintedWidth
            height: 5 * cropView.paintedHeight
            x: 0
            y: 0
            source: cropView.source
        }

        Image {
            id: crosshair
            anchors.fill: parent
            source: "images/circular218.png"
        }

    }
    Image {
        id: cropView
        anchors.top: parent.top
        anchors.left: zoomArea.right
        anchors.bottom: parent.bottom
        anchors.right: parent.right
        //source: "file:///G:/code/test/untitled1/test (1).jpg"

        CornerPoint {

            id: topLeft
            objectName: "topLeft"
            x: (parent.width - parent.paintedWidth) / 2 - this.width / 2
            y: (parent.height - parent.paintedHeight) / 2 - this.height / 2

            onXChanged: {
                zoom.x = cornerXRelativeToImg(x, topLeft);
                addCorner(topLeft);
                canvas.requestPaint();
                curPoint = objectName
            }
            onYChanged: {
                zoom.y = cornerYRelativeToImg(y, topLeft);
                addCorner(topLeft);
                canvas.requestPaint();
                curPoint = objectName
            }

        }

        CornerPoint {
            id: topRight
            objectName: "topRight"
            x: (parent.width - parent.paintedWidth) / 2 + parent.paintedWidth - this.width / 2
            y: (parent.height - parent.paintedHeight) / 2 - this.height / 2

            onXChanged: {
                zoom.x = cornerXRelativeToImg(x, topLeft);
                addCorner(topRight);
                canvas.requestPaint();
                curPoint = objectName
            }
            onYChanged: {
                zoom.y = cornerYRelativeToImg(y, topLeft);
                addCorner(topRight);
                canvas.requestPaint();
                curPoint = objectName
            }

        }

        CornerPoint {
            id: bottomLeft
            objectName: "bottomLeft"
            x: (parent.width - parent.paintedWidth) / 2 - this.width / 2
            y: (parent.height - parent.paintedHeight) / 2 + parent.paintedHeight - this.height / 2

            onXChanged: {
                zoom.x = cornerXRelativeToImg(x, topLeft);
                addCorner(bottomLeft);
                canvas.requestPaint();
                curPoint = objectName
            }
            onYChanged: {
                zoom.y = cornerYRelativeToImg(y, topLeft);
                addCorner(bottomLeft);
                canvas.requestPaint();
                curPoint = objectName
            }

        }

        CornerPoint {
            id: bottomRight
            objectName: "bottomRight"
            x: (parent.width - parent.paintedWidth) / 2 + parent.paintedWidth - this.width / 2
            y: (parent.height - parent.paintedHeight) / 2 + parent.paintedHeight - this.height / 2

            onXChanged: {
                zoom.x = cornerXRelativeToImg(x, topLeft);
                addCorner(bottomRight);
                canvas.requestPaint();
                curPoint = objectName
            }
            onYChanged: {
                zoom.y = cornerYRelativeToImg(y, topLeft);
                addCorner(bottomRight);
                canvas.requestPaint();
                curPoint = objectName
            }

        }

        Canvas {
            id: canvas
            anchors.fill: parent
            z: 10

            onPaint: {
                var context = getContext("2d");

                var offset = topLeft.width / 2;

                context.reset()
                context.beginPath();
                context.lineWidth = 2;
                context.moveTo(topLeft.x + offset, topLeft.y + offset);
                context.strokeStyle = "#87CEFA"

                context.lineTo(topRight.x + offset, topRight.y + offset);
                context.lineTo(bottomRight.x + offset, bottomRight.y + offset);
                context.lineTo(bottomLeft.x + offset, bottomLeft.y + offset);
                context.lineTo(topLeft.x + offset, topLeft.y + offset);
                context.closePath();
                context.stroke();
            }
        }
    }

    Button {
        id: button1
        anchors.top: zoom.bottom
        text: "ocr"
        onClicked: {
            crop.analyze(cropView.source, cropPoints);
        }
    }
    Button {
        id: button2
        anchors.top: button1.bottom
        text: "open"
        onClicked: {
            file.open()
        }
    }


}
function cornerXRelativeToImg(x, corner) {
    var transferred = (x - (cropView.width - cropView.paintedWidth) / 2 + corner.width / 2) * -5 + zoomArea.width / 2;
    return transferred;
}

function cornerYRelativeToImg(y, corner) {
    return (y - (cropView.height - cropView.paintedHeight) / 2  + corner.height / 2) * -5 + zoomArea.height / 2;
}

function addCorner(corner) {
    var offsetx = corner.width / 2;
    var offsety = corner.height / 2;
    var xScale = cropView.sourceSize.width / cropView.paintedWidth;
    var yScale = cropView.sourceSize.height / cropView.paintedHeight;
    cropPoints[corner.objectName] = Qt.point(xScale * (corner.x - (cropView.width - cropView.paintedWidth) / 2 + offsetx),
                                             yScale * (corner.y - (cropView.height - cropView.paintedHeight) / 2 + offsety));
}

}
