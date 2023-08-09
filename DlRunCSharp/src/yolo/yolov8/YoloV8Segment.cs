using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using DlRunCSharp;
using DlRunCSharp.src.export;
using HalconDotNet;
using OpenCvSharp;

namespace DlRunCSharp
{
    public class YoloV8Segment:Yolo
    {
        public YoloV8Segment(string path):base(path)
        {

        }

        protected override IntPtr CreatEngin(string path)
        {
            return Export.create_yolov8_segment(path);
        }

    }
}
