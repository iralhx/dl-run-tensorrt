using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DlRunCSharp
{
    public struct Box
    {
        public float x1 { get; set; }
        public float y1 { get; set; }
        public float x2 { get; set; }
        public float y2 { get; set; }
        public int label { get; set; }
        public IntPtr MatPtr { get; set;}


    }

}
