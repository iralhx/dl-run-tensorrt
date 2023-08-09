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
            string imgpath = @"I:\github\dl-run-tensorrt\1.jpg";
            string onnxpath = @"I:\github\dl-run-tensorrt/yolov5l_yuanzhu.onnx";
            string modelpath = @"I:\github\dl-run-tensorrt/yolov5l_yuanzhu.engine";

            if (!File.Exists( modelpath))
            {
                Trt.Onnx2Trt(onnxpath, modelpath);
            }

            Yolo yolo = new YoloV5Detetion(modelpath);
            double allTime = 0;
            for (int i = 0; i < 10000; i++)
            {
                HImage srcImage = new HImage(imgpath);
                //HImage srcImage1 = new HImage(imgpath1);
                //HOperatorSet.Rgb1ToGray(srcImage, out HObject grayImage);
                //Thread.Sleep(200);
                DateTime start = DateTime.Now;
                Box[] result1 = yolo.Forword(srcImage);

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
                srcImage.Dispose();
            }



            yolo.Dispose();

            Console.ReadKey();
        }
    }
}
