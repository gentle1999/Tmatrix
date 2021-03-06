/*matrix.h*/

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

/*释放矩阵所用内存*/
void free_mat(Matrix *mat);

/*设置矩阵元素，输入长度为row*column的一维数组，转化为矩阵*/
void set_mat(Matrix *mat, const double *data);

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