using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace DlRunCSharp
{
    public static class Cuda
    {
        public static int GetCudaRuntimeVersion()
        {
            return Export.getCudaRuntimeVersion();
        }
    }
}
