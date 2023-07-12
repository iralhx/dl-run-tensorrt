using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;

namespace TestDlCSharp
{
    internal class Program
    {
        [DllImport("dl-run-tensorrt.dll")]
        public static extern IntPtr create_findline(string path);
        [DllImport("dl-run-tensorrt.dll")]
        public static extern IntPtr findline_forwork(IntPtr ptr, IntPtr mat);
        [DllImport("dl-run-tensorrt.dll")]
        public static extern void dispose_findline(IntPtr ptr);

        static void Main(string[] args)
        {
            string imgpath = @"E:\VS\WorkSpa2022\dl-run-tensorrt/1.bmp";
            string modelpath = @"E:\VS\WorkSpa2022\dl-run-tensorrt/FullConModel.engine";
            Mat img = new Mat(imgpath,  ImreadModes.Grayscale);
            IntPtr ptr = create_findline(modelpath);
            IntPtr ptrimg = findline_forwork(ptr, img.CvPtr);
            Mat imgresult = new Mat(256,256, MatType.CV_32FC1, ptrimg);
            imgresult.SaveImage("./result.jpg");
            dispose_findline(ptr);
        }
    }
}
