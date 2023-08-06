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
            Mat img = new Mat(imgpath,ImreadModes.Color);
            HImage srcImage=new HImage(imgpath);
            HOperatorSet.Rgb1ToGray(srcImage, out HObject grayImage);

            for (int i = 0; i < 10000; i++)
            {
                Thread.Sleep(1000);
                DateTime start = DateTime.Now;
                Box[] result = yoloV8Segment.Forword(img);
                DateTime end = DateTime.Now;

                Console.WriteLine($"总时间time:{(end - start).TotalMilliseconds}ms");

            }


            //for (int i = 0; i < result.Length; i++)
            //{
            //    Box box = result[i];
            //    Mat mat = new Mat(box.MatPtr);
            //    HImage himg=new HImage("byte",mat.Width,mat.Height,mat.Data);
            //    HOperatorSet.ZoomImageSize(himg,out HObject zoomSeg, box.x2-box.x1, box.y2 - box.y1, "constant");
            //    HOperatorSet.Threshold(zoomSeg, out HObject region, 30, 255);
            //    HOperatorSet.MoveRegion(region, out region, box.y1, box.x1);

            //    HOperatorSet.OverpaintRegion(grayImage, region,box.label*20, "fill");

            //    HOperatorSet.WriteImage(himg, "jpg", 0, $"{i}himg.jpg");
            //}

            //HOperatorSet.WriteImage(grayImage, "jpg", 0, $"srchimg.jpg");

            yoloV8Segment.Dispose();

            //Console.ReadKey();
        }
    }
}
