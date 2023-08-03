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
using Te;

namespace TestDlCSharp
{
    internal class Program
    {
        [DllImport("teAiFlowEngine.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool TeAiFlowEngine_getAiNodeOutput(IntPtr pEngine, [MarshalAs(UnmanagedType.LPStr)] string pcNodeName, [MarshalAs(UnmanagedType.LPStr)] string pcFlowName, IntPtr pOutput, bool bBlocking = true);

        static void Main(string[] args)
        {
            Trt.SetDevice(0);
            string imgpath = @"I:\github\dl-run-tensorrt/bus1.jpg";
            string modelpath = @"I:\github\dl-run-tensorrt/yolov8n-seg.engine";

            YoloV8Segment yoloV8Segment = new YoloV8Segment(modelpath);
            Mat img = new Mat(imgpath,  ImreadModes.Color);
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
