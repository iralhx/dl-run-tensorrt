using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using DlRunCSharp;
using OpenCvSharp;

namespace TestDlCSharp
{
    internal class Program
    {
        public const string DllName = "I:\\github\\dl-run-tensorrt\\bin\\x64\\Debug\\dl-run-tensorrt.dll";


        [DllImport(DllName)]
        public static extern IntPtr create_findline(string path);
        static void Main(string[] args)
        {
            string imgpath = @"I:\github\dl-run-tensorrt/bus1.jpg";
            string modelpath = @"I:\github\dl-run-tensorrt/yolov8n-seg.engine";

            YoloV8Segment yoloV8Segment = new YoloV8Segment(modelpath);
            Mat img = new Mat(imgpath,  ImreadModes.Grayscale);
            Box[] result = yoloV8Segment.Forword(img);

            for (int i = 0; i < result.Length; i++)
            {
                Box box = result[i];
                Mat mat = new Mat(box.MatPtr);
                Cv2.ImShow("mat", mat);
                Console.ReadKey();
            }
            yoloV8Segment.Dispose();

            Console.ReadKey();
        }
    }
}
