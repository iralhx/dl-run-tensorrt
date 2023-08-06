using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace DlRunCSharp.src.export
{
    public class ExportHelper
    {
        public const string DllName = "dl-run-tensorrt.dll";


        [DllImport(DllName)]
        public static extern IntPtr himage_to_mat(IntPtr r, IntPtr g, IntPtr b, int height, int width);
    }
}
