using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
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
            IntPtr boxsPtr = IntPtr.Zero;
            int count = 0;
            Export.yolov8_forword(ptr, mat.CvPtr,ref boxsPtr,ref count);
            Box[] boxs = new Box[count];
            for (int i = 0; i < count; i++)
            {
                Box box = Export.get_vector_box(boxsPtr, i);
                boxs[i] = box;
            }
            return boxs;
        }

        public void Dispose()
        {
            Export.dispose(ptr);
        }

    }
}
