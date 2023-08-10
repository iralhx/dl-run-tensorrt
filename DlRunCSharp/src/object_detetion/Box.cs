using HalconDotNet;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DlRunCSharp
{
    public class Box
    {

        public float X1 { get; set; }
        public float Y1 { get; set; }
        public float X2 { get; set; }
        public float Y2 { get; set; }
        public float Confidence { get; set; }
        public int Label { get; set; }

        public HObject Region { get; set; }

        public Box()
        {

        }


        ~Box()
        {
            Dispose();
        }

        internal Box( YoloBox yoloBox)
        {
            this.X1=yoloBox.x1;
            this.Y1=yoloBox.y1;
            this.X2=yoloBox.x2;
            this.Y2=yoloBox.y2;
            this.Confidence=yoloBox.Confidence;
            this.Label=yoloBox.label;
            if (yoloBox.PointPtr==IntPtr.Zero)
            {
                return;
            }

            int count =Export.get_vector_point_size(yoloBox.PointPtr);
            if (count==0)
            {
                return;
            }

            float[] rows = new float[count];
            float[] cols = new float[count];
            Export.copy_vector_point(rows, cols, yoloBox.PointPtr);
            HTuple hrows= new HTuple(rows);
            HTuple hcols = new HTuple(cols);

            HObject xld, srcRegion;
            HOperatorSet.GenContourPolygonXld(out xld, hrows, hcols);
            HOperatorSet.GenRegionContourXld(xld, out srcRegion, "filled");

            //Mat mat = new Mat(yoloBox.MatPtr);
            //HImage himg = new HImage("byte", mat.Width, mat.Height, mat.Data);
            //HOperatorSet.ZoomImageSize(himg, out HObject zoomSeg, yoloBox.x2 - yoloBox.x1, yoloBox.y2 - yoloBox.y1, "constant");
            //HOperatorSet.Threshold(zoomSeg, out HObject srcRegion, 30, 255);
            HOperatorSet.MoveRegion(srcRegion, out HObject region, yoloBox.y1, yoloBox.x1);
            this.Region = region;

            //himg.Dispose();
            //zoomSeg.Dispose();
            srcRegion.Dispose();
        }


        public void Dispose()
        {
            Region?.Dispose();
        }

    }
}
