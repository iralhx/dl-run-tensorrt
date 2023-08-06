using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DlRunCSharp.src.trt
{
    public static class Trt
    {
        public static void SetDevice(int index)
        {
            Export.set_device(index);
        }



    }
}
