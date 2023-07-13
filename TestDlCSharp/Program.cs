using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;

namespace TestDlCSharp
{
    internal class Program
    {
        public const string dllName = "dl-run-tensorrt.dll";
        [DllImport(dllName)]
        public static extern IntPtr create_findline(string path);
        [DllImport(dllName)]
        public static extern IntPtr findline_forwork(IntPtr ptr, IntPtr mat);
        [DllImport(dllName)]
        public static extern void dispose_findline(IntPtr ptr);

        static void Main(string[] args)
        {

            string imgpath = @"I:\github\dl-run-tensorrt/1.bmp";
            string modelpath = @"I:\github\dl-run-tensorrt/FullConModel.engine";
            Mat img = new Mat(imgpath,  ImreadModes.Grayscale);
            IntPtr ptr = create_findline(modelpath);
            IntPtr ptrimg = findline_forwork(ptr, img.CvPtr);
            Mat imgresult = new Mat(256,256, MatType.CV_32FC1, ptrimg);
            imgresult.SaveImage("./result.jpg");
            dispose_findline(ptr);
            Console.ReadKey();
        }
    }
}
