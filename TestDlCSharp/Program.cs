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
            int version=Cuda.GetCudaRuntimeVersion();
            Console.WriteLine($"cuda version : {version}");
            

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
            for (int i = 0; i < 1000; i++)
            {
                HImage srcImage = new HImage(imgpath);
                DateTime start = DateTime.Now;
                Box[] result1 = yolo.Forword(srcImage);
                DateTime end = DateTime.Now;
                allTime += (end - start).TotalMilliseconds;
                Console.WriteLine($"当前索引：{i}，总时间{(end - start).TotalMilliseconds} 平均时间time:{allTime/(i+1)}ms");
                srcImage.Dispose();
            }

            yolo.Dispose();

            Console.ReadKey();
        }
    }
}
