﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace DlRunCSharp
{
    internal static class Export
    {
        public const string DllName = "dl-run-tensorrt.dll";


        [DllImport(DllName)]
        public static extern IntPtr create_findline(string path);
        [DllImport(DllName)]
        public static extern IntPtr create_yolov8_detetion(string path);
        [DllImport(DllName)]
        public static extern IntPtr create_yolov8_segment(string path);
        [DllImport(DllName)]
        public static extern IntPtr yolov8_forword(IntPtr ptr, IntPtr mat);


        [DllImport(DllName)]
        public static extern void dispose(IntPtr ptr);


        [DllImport(DllName)]
        public static extern int get_vector_box_size(IntPtr vector);

        [DllImport(DllName)]
        public static extern IntPtr get_vector_box(IntPtr vector,int index);

    }
}
