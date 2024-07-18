using DlRunCSharp.src.export;
using HalconDotNet;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Remoting.Channels;
using System.Text;
using System.Threading.Tasks;



namespace DlRunCSharp
{
    public class Segformer: Model<HImage>
    {

        public Segformer(string path):base(path)
        {

        }

        public override void Dispose()
        {
            Export.delete_segformer(this.Engin);
        }

        public override HImage Forword(Mat img)
        {
            IntPtr intPtr=Export.segformer_forword(this.Engin, img.Data);
            //假设我们知道，可以去C++获取
            return new HImage("real", 640, 640, intPtr);
        }


        public HImage Forword(HImage img)
        {

            int channels = img.CountChannels();
            IntPtr matPtr;
            if (channels == 1)
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
            IntPtr intPtr = Export.segformer_forword(this.Engin, matPtr);
            //假设我们知道，可以去C++获取
            return new HImage("real", 640, 640, intPtr);
        }


        protected override IntPtr CreatEngin(string path)
        {
            return Export.create_segformer(path);
        }

    }
}
