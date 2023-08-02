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
            IntPtr boxsPtr = Export.yolov8_forword(ptr, mat.CvPtr);
            int count = Export.get_vector_box_size(boxsPtr);
            Box[] boxs = new Box[count];
            for (int i = 0; i < count; i++)
            {
                IntPtr boxPtr = Export.get_vector_box(boxsPtr, i);
                Box box = Marshal.PtrToStructure<Box>(boxPtr);
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
