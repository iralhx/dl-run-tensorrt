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
using System.Windows.Forms;

namespace TestDlCSharp
{
    internal class Program
    {
        static void Main(string[] args)
        {
            int version=Cuda.GetCudaRuntimeVersion();
            Console.WriteLine($"cuda version : {version}");
            

            Trt.SetDevice(0);

            string rootPath = @"I:\Github\mmsegmentation\data\OuterCircle\train";

            string onnxpath = @"I:\github\dl-run-tensorrt/model/slide_outer_circle_Segformer.onnx";
            string modelpath = @"I:\github\dl-run-tensorrt/model/slide_outer_circle_Segformer.engine";

            if (!File.Exists( modelpath))
            {
                Trt.Onnx2Trt(onnxpath, modelpath);
            }
            ResultAnalysis analysis = new ResultAnalysis(new string[] {"凹坑","划痕" });
            Segformer segformer = new Segformer(modelpath);
            string imgPath = Path.Combine(rootPath, "image");
            string labelPath = Path.Combine(rootPath, "label");
            string savePath = Path.Combine(rootPath, "result");
            if (!Directory.Exists(savePath))
            {
                Directory.CreateDirectory(savePath);
            }
            string[] images =  Directory.GetFiles(imgPath);

            for (int i = 0; i < images.Length; i++)
            { 
                HWindowControl windowControl = new HWindowControl();
                HOperatorSet.SetPart(windowControl.HalconWindow, 0, 0, 640, 640);
                windowControl.Height = 640;
                windowControl.Width = 640;
                string name = Path.GetFileName(images[i]);
                string labelOnePath =Path.Combine(labelPath, name);
                HImage img = new HImage(images[i]);
                HImage labelImg = new HImage(labelOnePath);
                HOperatorSet.ZoomImageSize(img, out HObject objZoom, 640, 640, "constant");
                HOperatorSet.Compose3(objZoom, objZoom, objZoom, out HObject rgbImg);
                HImage rgbZoom=new HImage(rgbImg);
                HImage result = segformer.Forword(rgbZoom);
                rgbImg.Dispose(); 
                objZoom.Dispose();
                img.Dispose();
                HImage labelImgZoom =labelImg.ZoomImageSize(640, 640, "constant");
                windowControl.HalconWindow.DispImage(rgbZoom);
                analysis.Analysis(result, labelImgZoom, windowControl.HalconWindow);
                result.Dispose();
                labelImgZoom.Dispose();
                labelImg.Dispose();
                rgbZoom.Dispose();
                string saveImgPath = Path.Combine(savePath, name);
                HImage saveImg = windowControl.HalconWindow.DumpWindowImage();
                saveImg.GetImageSize(out HTuple a,out HTuple b);
                HOperatorSet.WriteImage(saveImg, "bmp",255, saveImgPath);
            }

            //double allTime = 0;
            //int allCount = 200;
            //int breakCount = 40;
            //for (int i = 0; i < allCount; i++)
            //{
            //    HImage srcImage = new HImage(imgpath);
            //    DateTime start = DateTime.Now;
            //    HImage img = segformer.Forword(srcImage);
            //    DateTime end = DateTime.Now;
            //    Console.WriteLine($"当前索引：{i}，单次时间 ：{(end - start).TotalMilliseconds}");
            //    srcImage.Dispose();
            //    img.Dispose();
            //    if (i<breakCount)
            //    {
            //        continue;
            //    }
            //    allTime += (end - start).TotalMilliseconds;
            //}

            //Console.WriteLine($"avg time : {allTime / (allCount - breakCount)}");

            segformer.Dispose();

            Console.WriteLine(analysis);
            Console.ReadKey();
        }

    }
}
