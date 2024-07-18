using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;



namespace DlRunCSharp
{
    public abstract class Model<T> : IMosel<T>,IDisposable
    {
        protected IntPtr m_engin;
        public IntPtr Engin { get{ return m_engin; }}
        public Model(string path)
        {
            m_engin = CreatEngin(path);
        }
        protected abstract IntPtr CreatEngin(string path);
        public abstract T Forword(Mat img);

        public abstract void Dispose();
    }
}
