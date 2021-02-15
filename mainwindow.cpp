#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <opencv2/opencv.hpp>
#include <fdeep/fdeep.hpp>
#include <QTimeLine>
#include <QTimer>

using namespace cv;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    //QTimeLine 3.7 секунды
    QTimeLine *timeLine = new QTimeLine(3700, this);
    timeLine->setFrameRange(0, 100);
    ui->progressBar->setValue(0);
//    connect(timeLine, SIGNAL(frameChanged(int)), ui->progressBar, SLOT(setValue(int)));
//    connect(ui->choose_image_button, SIGNAL(clicked()), timeLine, SLOT(start()));
    //по изменениям фреймов в таймлайне в прогресс бар устанавливаются значения
    connect(timeLine, &QTimeLine::frameChanged, ui->progressBar, &QProgressBar::setValue);
    //по нажатию кнопки предикта начинает заполнятся таймлайн (в прогресс бар летят оттуда значения)
    connect(ui->predict_button, &QPushButton::pressed, timeLine, &QTimeLine::start);






}

MainWindow::~MainWindow()
{
    delete ui;

}


void MainWindow::on_choose_image_button_clicked()
{
    QString image_file = QFileDialog::getOpenFileName(this,
         tr("Open Image"), "D:/Nadenenko/CProjects/QtProjects/fdeep/fdeep/", tr("Image Files (*.png *.jpg *.bmp *.tiff *.tif)"));
        ui->image_path->setText(image_file);
        std::string image_path = image_file.toStdString();
        cv::Mat img_inp;
        img_inp = cv::imread(image_path);
        ui->input_image_label->setPixmap(
                    QPixmap::fromImage(QImage(img_inp.data, img_inp.cols, img_inp.rows,
                                                           img_inp.step, QImage::Format_RGB888)));

}

void MainWindow::on_choose_model_button_clicked()
{
    QString model_file = QFileDialog::getOpenFileName(this,
         tr("Open Model"), "D:/Nadenenko/CProjects/QtProjects/fdeep/fdeep/", tr("Model Files (*.json)"));
        ui->model_path->setText(model_file);


}
//

void MainWindow::on_predict_button_clicked()
{


    const std::string i_p = ui->image_path->text().toStdString();
    cv::Mat img;
    img = cv::imread(i_p);
    const std::string m_p = ui->model_path->text().toStdString();

    const auto model = fdeep::load_model(m_p);

    const fdeep::tensor input =
            fdeep::tensor_from_bytes(img.ptr(),
                img.rows, img.cols, img.channels());
    const auto result = model.predict({ input });
    const auto single = fdeep::internal::single_tensor_from_tensors(result);
    const cv::Mat image3(
            cv::Size(input.shape().width_, input.shape().height_), CV_32FC1);
    const auto values = single.to_vector();
    std::memcpy(image3.data, values.data(), values.size() * sizeof(float));
    cv::Mat tempImage5;
    cv::Mat image5;
    cv::normalize(image3, tempImage5, 255.0, 0.0, cv::NORM_MINMAX);
    tempImage5.convertTo(image5, CV_8UC3);

    ui->output_manual_image_label->setPixmap(
                QPixmap::fromImage(QImage(image5.data, image5.cols, image5.rows,
                                                       image5.step, QImage::Format_Indexed8)));

    cv::Mat thresholded;
        threshold(image5, thresholded, 100, 255, THRESH_BINARY);

    ui->output_auto_image_label->setPixmap(
                QPixmap::fromImage(QImage(thresholded.data, thresholded.cols, thresholded.rows,
                                                       thresholded.step, QImage::Format_Indexed8)));

}

