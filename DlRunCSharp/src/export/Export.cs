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
        public static extern void set_device(int index);

        [DllImport(DllName)]
        public static extern bool onnx2trt( string onnxPath, string trtPath);

        [DllImport(DllName)]
        public static extern IntPtr create_findline(string path);
        [DllImport(DllName)]
        public static extern IntPtr create_yolov8_detetion(string path); 
        [DllImport(DllName)]
        public static extern IntPtr create_yolov5_detetion(string path); 
        [DllImport(DllName)]
        public static extern IntPtr create_yolov8_segment(string path);
        [DllImport(DllName)]
        public static extern IntPtr yolo_forword(IntPtr ptr, IntPtr mat, ref int size);


        [DllImport(DllName)]
        public static extern void dispose(IntPtr ptr);


        [DllImport(DllName)]
        public static extern int get_vector_box_size(IntPtr vector);

        [DllImport(DllName)]
        public static extern IntPtr get_vector_box(IntPtr vector,int index);


        [DllImport(DllName)]
        public static extern void delete_vector_box(IntPtr vector);



        [DllImport(DllName)]
        public static extern int get_vector_point_size(IntPtr points);

        [DllImport(DllName)]
        public static extern void copy_vector_point([Out][MarshalAs(UnmanagedType.LPArray)] float[]  rows, [Out][MarshalAs(UnmanagedType.LPArray)] float[] cols, IntPtr points);


        [DllImport(DllName)]
        public static extern int getCudaRuntimeVersion();


        /// <summary>
        /// 创建模型
        /// </summary>
        [DllImport(DllName)]
        public static extern IntPtr create_segformer(string path);


        /// <summary>
        /// 进行预测
        /// 返回值为float*
        /// </summary>
        [DllImport(DllName)]
        public static extern IntPtr segformer_forword(IntPtr model,IntPtr img);

        /// <summary>
        /// 释放
        /// </summary>
        [DllImport(DllName)]
        public static extern void delete_segformer(IntPtr model);

    }
}
