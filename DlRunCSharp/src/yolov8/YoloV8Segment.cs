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
    public class YoloV8Segment
    {
        private IntPtr ptr;

        public YoloV8Segment(string path)
        {
            ptr = Export.create_yolov8_segment(path);
        }

        public Box[] Forword(Mat mat)
        {
            int count = 0;
            IntPtr result =  Export.yolov8_forword(ptr, mat.CvPtr, ref count);
            Box[] boxs = new Box[count];
            YoloBox yoloBox;
            for (int i = 0; i < count; i++)
            {
                yoloBox = Export.get_vector_box(result, i);
                boxs[i] = new Box(yoloBox);
            }
            Export.delete_vector_box(result);

            return boxs;
        }

        public Box[] Forword(HImage img)
        {

            HOperatorSet.GetImagePointer3(img, out HTuple imgRPtr, out HTuple imgGPtr, out HTuple imgBPtr,
                out HTuple type1, out HTuple width, out HTuple height);

           IntPtr matPtr = ExportHelper.himage_to_mat(imgRPtr.IP, imgGPtr.IP, imgBPtr.IP, height, width);
            //BGR
            Mat mat = new Mat(matPtr);
            Box[] boxs = Forword(mat);
            return boxs;
        }

        public void Dispose()
        {
            Export.dispose(ptr);
        }

    }
}
