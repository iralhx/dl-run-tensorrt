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
            string imgpath = "I:\\github\\infer\\workspace\\inference\\gril.jpg";
            string onnxpath = @"I:\github\dl-run-tensorrt/yolov8l-seg.onnx";
            string modelpath = @"I:\github\dl-run-tensorrt/yolov8l-seg.engine";

            if (!File.Exists( modelpath))
            {
                Trt.Onnx2Trt(onnxpath, modelpath);
            }

            string imgpath1 = @"I:\\github\\dl-run-tensorrt\bus1.jpg";
            YoloV8Segment yoloV8Segment = new YoloV8Segment(modelpath);
            double allTime = 0;
            for (int i = 0; i < 10000; i++)
            {
                Mat img = new Mat(imgpath, ImreadModes.Color);
                Mat img1 = new Mat(imgpath1, ImreadModes.Color);
                //HImage srcImage = new HImage(imgpath);
                //HImage srcImage1 = new HImage(imgpath1);
                //HOperatorSet.Rgb1ToGray(srcImage, out HObject grayImage);
                Thread.Sleep(200);
                DateTime start = DateTime.Now;
                Box[] result = yoloV8Segment.Forword(img1);
                Box[] result1 = yoloV8Segment.Forword(img);

                //for (int j = 0; j < result.Length; j++)
                //{
                //    HOperatorSet.OverpaintRegion(grayImage, result[j].Region, result[j].Label * 20, "fill");
                //    //HOperatorSet.WriteRegion(result[j].Region, $"I:\\github\\dl-run-tensorrt\\{i}.hobj");
                //}
                //HOperatorSet.WriteImage(grayImage, "png", 0, $"I:\\github\\dl-run-tensorrt\\{i}.png");
                DateTime end = DateTime.Now;
                //img.Dispose();
                allTime += (end - start).TotalMilliseconds;
                Console.WriteLine($"当前索引：{i}，总时间{(end - start).TotalMilliseconds} 平均时间time:{allTime/(i+1)}ms");
                //srcImage.Dispose();
            }



            yoloV8Segment.Dispose();

            Console.ReadKey();
        }
    }
}
