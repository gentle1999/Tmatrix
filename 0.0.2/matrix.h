/*
matrix.h
version:0.0.2
author TMJ
*/

typedef struct matrix
{
    unsigned int row;
    unsigned int column;
    double **data;
} Matrix;

/*矩阵初始化为全0*/
void clear_mat(Matrix *mat);

/*创建新矩阵*/
Matrix create_mat(unsigned int row, unsigned int column);

/*创建全为1的矩阵*/
Matrix ones_mat(int row, int column);

/*释放矩阵所用内存*/
void free_mat(Matrix *mat);

/*设置矩阵元素，输入长度为row*column的一维数组，转化为矩阵*/
void set_mat(Matrix *mat, const double *data);

/*截取矩阵中的一部分，输入矩阵地址以及四角坐标，(x1,x2,y1,y2)，从左上角(0,0)开始到(x,y)范围内元素，x为横坐标，y为纵坐标*/
Matrix interception_mat(const Matrix *mat, int x1, int x2, int y1, int y2);

/*拼接矩阵，模式1、2分别代表后一矩阵拼接到前一矩阵的下、右，必须满足长度匹配*/
Matrix joint_mat(const Matrix *a, const Matrix *b, int mode);

/*矩阵加法*/
Matrix plus_mat(const Matrix *mat1, const Matrix *mat2);

/*矩阵减法*/
Matrix minus_mat(const Matrix *mat1, const Matrix *mat2);

/*矩阵数乘*/
Matrix scalar_mult_mat(const Matrix *mat, const double k);

/*矩阵转置*/
Matrix transpose_mat(const Matrix *mat);

/*矩阵乘法*/
Matrix mult_mat(const Matrix *mat1, const Matrix *mat2);

/*矩阵行列式计算*/
double det_mat(const Matrix *mat);

/*复制矩阵*/
Matrix copy_mat(const Matrix *mat0);

/*创建单位矩阵*/
Matrix E_mat(const unsigned int row);

/*矩阵求逆*/
Matrix inv_mat(const Matrix *m);

/*展示矩阵内容，输入矩阵的地址和矩阵的名称*/
void show_mat(const Matrix *mat, char *name);

/*LUP分解，将输入矩阵分解成L、U、P三个矩阵，需要在函数外先创建对应的矩阵，其中L、U为空，P为单位矩阵，LU=PA，P^(-1)LU=A*/
void LUP_mat(const Matrix *m, Matrix *L, Matrix *U, Matrix *P);

/*高斯消元法（利用LUP分解）解线性方程组，输入方阵地址及列向量地址*/
Matrix Gauss_mat(const Matrix *mat, const Matrix *x);

/*幂法求按模最大的特征值，输入矩阵的地址以及迭代次数*/
double power_mat(const Matrix *mat, unsigned int t);

/*反幂法求按模最小的特征值，输入矩阵的地址以及迭代次数*/
double anti_power_mat(const Matrix *mat, unsigned int t);

/*求矩阵的谱半径，输入矩阵的地址以及迭代次数*/
double radius_mat(const Matrix *mat, unsigned int t);

/*求矩阵的1范数（列模）*/
double norm_1_mat(const Matrix *mat);

/*求矩阵的2范数（谱模），输入矩阵的地址以及迭代次数*/
double norm_2_mat(const Matrix *mat, unsigned int t);

/*求矩阵的无穷范数（行模）*/
double norm_inf_mat(const Matrix *mat);

/*求矩阵的1条件数*/
double cond_1_mat(const Matrix *mat);

/*求矩阵的2条件数，需要迭代次数*/
double cond_2_mat(const Matrix *mat, unsigned int t);

/*求矩阵的无穷条件数*/
double cond_inf_mat(const Matrix *mat);

/*创建希尔伯特矩阵*/
Matrix Hilbert_mat(const unsigned int n);

/*Jacobi迭代法解线性方程组，输入系数矩阵地址以及列向量地址，以及迭代次数，方阵A非奇异且对角线元素不能为零，A=D+L+U（D为对角，L为下三角，U为上三角），迭代公式 x=D^(-1)[-(L+U)x+b]*/
Matrix Jacobi_mat(const Matrix *mat, const Matrix *b, unsigned int t);

/*Gauss-Seidel迭代法解线性方程组，输入系数矩阵地址以及列向量地址，以及迭代次数，方阵A非奇异且对角线元素不能为零，A=D+L+U（D为对角，L为下三角，U为上三角），迭代公式 x=(D+L)^(-1)(-Ux+b)*/
Matrix Guass_seidel_mat(const Matrix *mat, const Matrix *b, unsigned int t);

/*拉格朗日插值法进行多项式回归，输入两个列向量分别保存有序的(x_0,y_0)值对，以及需要验证的变量x*/
double Lagrange_Interpolation(const Matrix *x, const Matrix *y, double x_);

/*计算拉格朗日插值法中的重心权，输入列向量形式有序的x_0*/
Matrix weight_gravity(const Matrix *x);

/*计算拉格朗日插值法中的l(x)，输入列向量形式有序的x_0，以及变量x*/
double Lagrange_lx(const Matrix *x, double x_);

/*重心拉格朗日插值法（第一型）进行多项式回归，输入两个列向量分别保存有序的(x_0,y_0)值对，以及需要验证的变量x*/
double Lagrange_Gravity_1_Interpolation(const Matrix *x, const Matrix *y, double x_);

/*重心拉格朗日插值法（第二型）进行多项式回归，输入两个列向量分别保存有序的(x_0,y_0)值对，以及需要验证的变量x*/
double Lagrange_Gravity_2_Interpolation(const Matrix *x, const Matrix *y, double x_);

/*计算牛顿插值法需要的均差*/
Matrix inequality_mat(const Matrix *x, const Matrix *y, int d);

/*牛顿插值法进行多项式回归，输入两个列向量分别保存有序的(x_0,y_0)值对，以及需要验证的变量x*/
double Newton_Interpolation(const Matrix *x, const Matrix *y, double x_);

/*计算Hermite插值法需要的l(x)，输入列向量形式有序的x，变量x_，以及参数i*/
double Hermite_lx(const Matrix *x, double x_, int i);

/*计算Hermite插值法需要的l'(x)，输入列向量形式有序的x，以及参数i*/
double Hermite_lx_(const Matrix *x, int i);

/*二次Hermite插值法，输入三个列向量x,y,y'，以及需要验证的变量x_*/
double Hermite_2_Interpolation(const Matrix *x, const Matrix *y, const Matrix *y_, double x_);

/*计算步长，输入列向量，返回长度减一的向量存储步长*/
Matrix step_length(const Matrix *x);

/*三次样条插值（自由边界），输入两个列向量x,y，生成一个存储了四组系数的矩阵*/
Matrix spline_natural(const Matrix *x, const Matrix *y);

/*三次样条插值（固定边界），输入两个列向量x,y，给定两端点处微分值A、B，生成一个存储了四组系数的矩阵*/
Matrix spline_clamped(const Matrix *x, const Matrix *y, double A, double B);

/*三次样条插值（非节点边界），输入两个列向量x,y，生成一个存储了四组系数的矩阵*/
Matrix spline_NAK(const Matrix *x, const Matrix *y);

/*利用输入的样条插值矩阵进行计算，输入样例点向量和变量x_，给出拟合得到的函数值*/
double spline_out(const Matrix *m, const Matrix *x, double x_);

/*最小二乘法进行多元线性回归，输出一组列向量存储系数*/
Matrix least_square(const Matrix *x, const Matrix *y);

/*根据给出的系数向量，计算线性函数值*/
double linear_value(const Matrix *w, const Matrix *x);

/*计算误差平方和*/
double SSE(const Matrix *x_, const Matrix *y, const Matrix *w);

/*计算总修正平均值*/
double SST(const Matrix *x_, const Matrix *y, const Matrix *w);

/*计算决定系数R^2*/
double R_square(const Matrix *x_, const Matrix *y, const Matrix *w);