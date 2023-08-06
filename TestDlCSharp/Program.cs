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
            string imgpath = @"I:\github\dl-run-tensorrt/bus1.jpg";
            string modelpath = @"I:\github\dl-run-tensorrt/yolov8n-seg.engine";

            YoloV8Segment yoloV8Segment = new YoloV8Segment(modelpath);

            for (int i = 0; i < 10000; i++)
            {
                Mat img = new Mat(imgpath, ImreadModes.Color);
               // HImage srcImage = new HImage(imgpath);
                //Thread.Sleep(1000);
                DateTime start = DateTime.Now;
                Box[] result = yoloV8Segment.Forword(img);
                DateTime end = DateTime.Now;

                Console.WriteLine($"总时间time:{(end - start).TotalMilliseconds}ms");
                //srcImage.Dispose();
            }


            yoloV8Segment.Dispose();

            //Console.ReadKey();
        }
    }
}
