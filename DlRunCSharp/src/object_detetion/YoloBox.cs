﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace DlRunCSharp
{
    [StructLayout( LayoutKind.Sequential)]
    public struct YoloBox
    {
        public float x1 { get; set; }
        public float y1 { get; set; }
        public float x2 { get; set; }
        public float y2 { get; set; }
        public float Confidence { get; set; }
        public int label { get; set; }
        public IntPtr PointPtr { get; set;}


    }

}
