using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


#region 文件信息
/****************************************************************
*	公司名称：福建力和行智能科技有限公司
*   作    者：Ira
*   CLR 版本：4.0.30319.42000
*   创建时间：2024/5/28 17:22:57
*	用 户 名：zzhan 
*   描述说明：
*
*   修改历史：
*		1）	修 改 人：
*			修改日期：
*			修改内容：			
*
*****************************************************************/
#endregion

namespace DlRunCSharp
{
    /// <summary>
    /// 类    名:YoloV8Detetion
    /// 描    述:
    /// 修改时间:2024/5/28 17:22:57
    /// </summary>
    public class YoloV8Detetion:Yolo
    {
        #region 成员变量

        #region private



        #endregion


        #region protected



        #endregion


        #region public


        #endregion

        #endregion


        #region 构造函数
        /// <summary>
        /// 函 数 名:构造函数
        /// 函数描述:默认构造函数
        /// 修改时间:2024/5/28 17:22:57
        /// </summary>
        public YoloV8Detetion( string path):base(path)
        {

        }

        protected override IntPtr CreatEngin(string path)
        {
            return Export.create_yolov8_detetion(path);
        }
        #endregion

        #region 父类函数重载、接口实现

        #endregion

        #region 函数

        #region private



        #endregion


        #region protected



        #endregion


        #region public


        #endregion

        #endregion
    }
}
