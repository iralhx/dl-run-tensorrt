using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace DlRunCSharp
{

    public interface IMosel<T>
    {
        /// <summary>
        /// TRT的引擎文件指针
        /// </summary>
        IntPtr Engin { get; }
        /// <summary>
        /// 推理
        /// </summary>
        T Forword(Mat img);

    }
}
