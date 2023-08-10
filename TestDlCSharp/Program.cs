using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using DlRunCSharp;
using DlRunCSharp.src.trt;
using OpenCvSharp;
using HalconDotNet;
using System.Threading;

namespace TestDlCSharp
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Trt.SetDevice(0);
            string imgpath = @"I:\github\dl-run-tensorrt\bus1.jpg";
            string onnxpath = @"I:\github\dl-run-tensorrt/model/yolov8l-seg.onnx";
            string modelpath = @"I:\github\dl-run-tensorrt/model/yolov8l-seg.engine";

            if (!File.Exists( modelpath))
            {
                Trt.Onnx2Trt(onnxpath, modelpath);
            }

            Yolo yolo = new YoloV8Segment(modelpath);
            double allTime = 0;
            for (int i = 0; i < 10000; i++)
            {
                HImage srcImage = new HImage(imgpath);
                //HImage srcImage1 = new HImage(imgpath1);
                //HOperatorSet.Rgb1ToGray(srcImage, out HObject grayImage);
                //Thread.Sleep(200);
                DateTime start = DateTime.Now;
                Box[] result1 = yolo.Forword(srcImage);
                //foreach (Box item in result1)
                //{
                //    HOperatorSet.GenRectangle1(out HObject region, item.Y1, item.X1, item.Y2,  item.X2);
                //    HOperatorSet.GenContourRegionXld(region, out HObject xld, "border");
                //    HOperatorSet.GetContourXld(xld, out HTuple rows, out HTuple cols);
                //    HOperatorSet.GenRegionPolygon(out HObject xldRegion, rows, cols);
                //    HOperatorSet.OverpaintRegion(grayImage, xldRegion, 255, "fill");
                //}
                //HOperatorSet.WriteImage(grayImage, "png", 0, $"I:\\github\\dl-run-tensorrt\\adsdasd{i}.png");
                DateTime end = DateTime.Now;
                //img.Dispose();
                allTime += (end - start).TotalMilliseconds;
                Console.WriteLine($"当前索引：{i}，总时间{(end - start).TotalMilliseconds} 平均时间time:{allTime/(i+1)}ms");
                srcImage.Dispose();
            }



            yolo.Dispose();

            Console.ReadKey();
        }
    }
}
