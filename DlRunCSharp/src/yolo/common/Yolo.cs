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
    public abstract class Yolo
    {

        protected IntPtr m_engin;
        public Yolo(string path)
        {
            m_engin=CreatEngin(path);
        }

        protected abstract IntPtr CreatEngin(string path);


        public Box[] Forword(Mat mat)
        {
            return Forword(mat.CvPtr);
        }

        public Box[] Forword(IntPtr mat)
        {
            int count = 0;
            IntPtr result = Export.yolo_forword(m_engin, mat, ref count);
            Box[] boxs = new Box[count];
            for (int i = 0; i < count; i++)
            {
                IntPtr boxPtr = Export.get_vector_box(result, i);
                YoloBox yoloBox = Marshal.PtrToStructure<YoloBox>(boxPtr);
                boxs[i] = new Box(yoloBox);
            }
            GC.KeepAlive(result);
            Export.delete_vector_box(result);

            return boxs;
        }

        public Box[] Forword(HImage img)
        {
            int channels = img.CountChannels();
            
            IntPtr matPtr;
            if (channels==1)
            {
                HOperatorSet.GetImagePointer1(img, out HTuple imgPtr,
                    out HTuple type1, out HTuple width, out HTuple height);
                matPtr = ExportHelper.himage_to_mat(imgPtr.IP, imgPtr.IP, imgPtr.IP, height, width);
            }
            else if (channels == 3)
            {

                HOperatorSet.GetImagePointer3(img, out HTuple imgRPtr, out HTuple imgGPtr, out HTuple imgBPtr,
                    out HTuple type1, out HTuple width, out HTuple height);
                matPtr = ExportHelper.himage_to_mat(imgRPtr.IP, imgGPtr.IP, imgBPtr.IP, height, width);
            }
            else
            {
                throw new Exception("不支持的图片格式");
            }

            //BGR
            Box[] boxs = Forword(matPtr);
            return boxs;
        }

        public void Dispose()
        {
            Export.dispose(m_engin);
        }

    }
}
