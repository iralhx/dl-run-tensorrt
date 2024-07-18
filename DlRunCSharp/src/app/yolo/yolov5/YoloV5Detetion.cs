using DlRunCSharp.src.export;
using HalconDotNet;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;


namespace DlRunCSharp
{
    public class YoloV5Detetion:Yolo
    {
        public YoloV5Detetion(string path):base(path) 
        {

        }

        protected override IntPtr CreatEngin(string path)
        {
            return Export.create_yolov5_detetion(path);
        }
    }
}
