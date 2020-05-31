/*
matrix.c
version:0.0.3
author TMJ
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define ZERO 1e-13
#define ABS(x)                            \
    ({                                    \
        typeof(x) __tmp_x = (x);          \
        __tmp_x < 0 ? -__tmp_x : __tmp_x; \
    })
#define EXCHANGE(a, b)           \
    {                            \
        typeof(a) __tmp_a = (a); \
        a = b;                   \
        b = __tmp_a;             \
    }

/*定义矩阵类型*/
typedef struct matrix
{
    unsigned int row;
    unsigned int column;
    double **data;
} Matrix;

/*矩阵初始化为全0*/
void clear_mat(Matrix *mat)
{
    unsigned int i, j;
    for (i = 0; i < mat->row; i++)
    {
        for (j = 0; j < mat->column; j++)
        {
            mat->data[i][j] = 0;
        }
    }
}

/*展示矩阵内容，输入矩阵的地址和矩阵的名称*/
void show_mat(const Matrix *mat, char *name)
{
    unsigned int i, j;
    printf("%s=\n", name);
    for (i = 0; i < mat->row; i++)
    {
        for (j = 0; j < mat->column; j++)
        {
            printf("%8.4lf", mat->data[i][j]);
        }
        printf("\n");
    }
}

/*创建新矩阵*/
Matrix create_mat(unsigned int row, unsigned int column)
{
    Matrix mat;
    if (row <= 0 || column <= 0)
    {
        printf("error, row or column cannot lower than 1\n");
        exit(1);
    }
    else
    {
        mat.row = row;
        mat.column = column;
        mat.data = (double **)malloc(row * sizeof(double *)); //先分配指针的指针
        if (mat.data == NULL)
        {
            printf("error, data == NULL\n");
            exit(1);
        }
        unsigned int i;
        for (i = 0; i < row; i++)
        {
            *(mat.data + i) = (double *)malloc(column * sizeof(double)); //分配每行的指针
            if (mat.data[i] == NULL)
            {
                printf("error, data == NULL\n");
                exit(1);
            }
        }
        clear_mat(&mat);
    }
    return mat;
}

/*创建全为1的矩阵*/
Matrix ones_mat(int row, int column)
{
    Matrix ones = create_mat(row, column);
    int i, j;
    for (i = 0; i < row; i++)
    {
        for (j = 0; j < column; j++)
        {
            ones.data[i][j] = 1;
        }
    }
    return ones;
}

/*截取矩阵中的一部分，输入矩阵地址以及四角坐标，(x1,x2,y1,y2)，从左上角(0,0)开始到(x,y)范围内元素，x为横坐标，y为纵坐标*/
Matrix interception_mat(const Matrix *mat, int x1, int x2, int y1, int y2)
{
    if (x1 < 0 || y1 < 0 || x2 > mat->column - 1 || y2 > mat->row - 1)
    {
        printf("error, Overstep");
        exit(1);
    }
    Matrix m = create_mat(y2 - y1 + 1, x2 - x1 + 1);
    int i, j;
    for (i = x1; i <= x2; i++)
    {
        for (j = y1; j <= y2; j++)
        {
            m.data[j][i] = mat->data[j][i];
        }
    }
    return m;
}

/*拼接矩阵，模式1、2分别代表后一矩阵拼接到前一矩阵的下、右，必须满足长度匹配*/
Matrix joint_mat(const Matrix *a, const Matrix *b, int mode)
{
    switch (mode)
    {
    case 1:
    {
        if (a->column != b->column)
        {
            printf("error, column is mismatch");
            exit(1);
        }
        Matrix ans = create_mat(a->row + b->row, a->column);
        int i, j;
        for (i = 0; i < a->row; i++)
        {
            for (j = 0; j < a->column; j++)
            {
                ans.data[i][j] = a->data[i][j];
            }
        }
        for (i = a->row; i < a->row + b->row; i++)
        {
            for (j = 0; j < a->column; j++)
            {
                ans.data[i][j] = b->data[i - a->row][j];
            }
        }
        return ans;
    }
    case 2:
    {
        if (a->row != b->row)
        {
            printf("error, row is mismatch");
            exit(1);
        }
        Matrix ans = create_mat(a->row, a->column + b->column);
        int i, j;
        for (i = 0; i < a->row; i++)
        {
            for (j = 0; j < a->column; j++)
            {
                ans.data[i][j] = a->data[i][j];
            }
            for (j = a->column; j < a->column + b->column; j++)
            {
                ans.data[i][j] = b->data[i][j - a->column];
            }
        }
        return ans;
    }
    default:
    {
        printf("error, mode is wrong");
        exit(1);
    }
    }
}

/*释放矩阵所用内存*/
void free_mat(Matrix *mat)
{
    unsigned int i;
    for (i = 0; i < mat->row; i++)
    {
        free(mat->data[i]); //释放行
    }
    free(mat->data); //释放头指针
}

/*设置矩阵元素，输入长度为row*column的一维数组，转化为矩阵*/
void set_mat(Matrix *mat, const double *data)
{
    unsigned int i, j;
    for (i = 0; i < mat->row; i++)
    {
        for (j = 0; j < mat->column; j++)
        {
            mat->data[i][j] = data[i * mat->column + j];
        }
    }
}

/*矩阵加法*/
Matrix plus_mat(const Matrix *mat1, const Matrix *mat2)
{
    if (mat1->row != mat2->row)
    {
        printf("error, rows are not equal\n");
        exit(1);
    }
    if (mat1->column != mat2->column)
    {
        printf("error, columns are not equal\n");
        exit(1);
    }
    Matrix mat;
    unsigned int i, j;
    mat = create_mat(mat1->row, mat1->column);
    for (i = 0; i < mat1->row; i++)
    {
        for (j = 0; j < mat1->column; j++)
        {
            mat.data[i][j] = mat1->data[i][j] + mat2->data[i][j];
        }
    }
    return mat;
}

/*矩阵减法*/
Matrix minus_mat(const Matrix *mat1, const Matrix *mat2)
{
    if (mat1->row != mat2->row)
    {
        printf("error, rows are not equal\n");
        exit(1);
    }
    if (mat1->column != mat2->column)
    {
        printf("error, columns are not equal\n");
        exit(1);
    }
    Matrix mat;
    unsigned int i, j;
    mat = create_mat(mat1->row, mat1->column);
    for (i = 0; i < mat1->row; i++)
    {
        for (j = 0; j < mat1->column; j++)
        {
            mat.data[i][j] = mat1->data[i][j] - mat2->data[i][j];
        }
    }
    return mat;
}

/*矩阵加法，允许将一个较小的矩阵（后者）加到一个较大的矩阵（前者）的左上角*/
Matrix plus_mat_2(const Matrix *mat1, const Matrix *mat2)
{
    if (mat1->row < mat2->row)
    {
        printf("error\n");
        exit(1);
    }
    if (mat1->column < mat2->column)
    {
        printf("error\n");
        exit(1);
    }
    Matrix mat = create_mat(mat1->row, mat1->column);
    unsigned int i, j;
    for (i = 0; i < mat1->row; i++)
    {
        for (j = 0; j < mat1->column; j++)
        {
            mat.data[i][j] = mat1->data[i][j];
        }
    }
    for (i = 0; i < mat2->row; i++)
    {
        for (j = 0; j < mat2->column; j++)
        {
            mat.data[i][j] += mat2->data[i][j];
        }
    }
    return mat;
}

/*矩阵减法，允许将一个较小的矩阵（后者）减到一个较大的矩阵（前者）的左上角*/
Matrix minus_mat_2(const Matrix *mat1, const Matrix *mat2)
{
    if (mat1->row < mat2->row)
    {
        printf("error\n");
        exit(1);
    }
    if (mat1->column < mat2->column)
    {
        printf("error\n");
        exit(1);
    }
    Matrix mat = create_mat(mat1->row, mat1->column);
    unsigned int i, j;
    for (i = 0; i < mat1->row; i++)
    {
        for (j = 0; j < mat1->column; j++)
        {
            mat.data[i][j] = mat1->data[i][j];
        }
    }
    for (i = 0; i < mat2->row; i++)
    {
        for (j = 0; j < mat1->column; j++)
        {
            mat.data[i][j] -= mat2->data[i][j];
        }
    }
    return mat;
}

/*矩阵数乘*/
Matrix scalar_mult_mat(const Matrix *mat0, const double k)
{
    Matrix mat;
    unsigned int i, j;
    mat = create_mat(mat0->row, mat0->column);
    for (i = 0; i < mat0->row; i++)
    {
        for (j = 0; j < mat0->column; j++)
        {
            mat.data[i][j] = k * mat0->data[i][j];
        }
    }
    return mat;
}

/*矩阵转置*/
Matrix transpose_mat(const Matrix *mat)
{
    Matrix mat_t;
    mat_t = create_mat(mat->column, mat->row);
    unsigned int i, j;
    for (i = 0; i < mat->row; i++)
    {
        for (j = 0; j < mat->column; j++)
        {
            mat_t.data[j][i] = mat->data[i][j];
        }
    }
    return mat_t;
}

/*矩阵乘法*/
Matrix mult_mat(const Matrix *mat1, const Matrix *mat2)
{
    Matrix mat;
    if (mat1->column != mat2->row)
    {
        printf("error, mat1's column is not equal to mat2's row\n");
        exit(1);
    }
    else
    {
        mat = create_mat(mat1->row, mat2->column);
        unsigned int i, j, k;
        for (i = 0; i < mat1->row; i++)
        {
            for (j = 0; j < mat2->column; j++)
            {
                for (k = 0; k < mat1->column; k++)
                {
                    mat.data[i][j] += mat1->data[i][k] * mat2->data[k][j];
                }
            }
        }
    }
    return mat;
}

/*复制矩阵*/
Matrix copy_mat(const Matrix *mat0)
{
    Matrix mat;
    unsigned int i, j;
    mat = create_mat(mat0->row, mat0->column);
    for (i = 0; i < mat0->row; i++)
    {
        for (j = 0; j < mat0->column; j++)
        {
            mat.data[i][j] = mat0->data[i][j];
        }
    }
    return mat;
}

/*计算矩阵的Kroneck积*/
Matrix kroneck_mat(const Matrix *a, const Matrix *b)
{
    Matrix ans = create_mat(a->row * b->row, a->column * b->column);
    int m, n, p, q;
    for (m = 0; m < a->row; m++)
    {
        for (n = 0; n < a->column; n++)
        {
            for (p = 0; p < b->row; p++)
            {
                for (q = 0; q < b->column; q++)
                {
                    ans.data[m * b->row + p][n * b->column + q] = a->data[m][n] * b->data[p][q];
                }
            }
        }
    }
    return ans;
}

/*计算矩阵的Hadamard积，要求输入的矩阵大小相同*/
Matrix hadamard_mat(const Matrix *a, const Matrix *b)
{
    if (a->row != b->row)
    {
        printf("error, rows are not equal\n");
        exit(1);
    }
    if (a->column != b->column)
    {
        printf("error, columns are not equal\n");
        exit(1);
    }
    Matrix ans = create_mat(a->row, a->column);
    int i, j;
    for (i = 0; i < a->row; i++)
    {
        for (j = 0; j < a->column; j++)
        {
            ans.data[i][j] = a->data[i][j] * b->data[i][j];
        }
    }
    return ans;
}

/*创建单位矩阵*/
Matrix E_mat(const unsigned int row)
{
    if (row <= 0)
    {
        printf("error, row cannot lower than 1\n");
        exit(1);
    }
    Matrix mat;
    unsigned int i;
    mat = create_mat(row, row);
    for (i = 0; i < row; i++)
    {
        mat.data[i][i] = 1;
    }
    return mat;
}

/*求向量的平均值*/
double average_vector(const Matrix *mat)
{
    if (mat->row != 1 && mat->column != 1)
    {
        printf("error, not vector");
        exit(1);
    }
    long len;
    int flag = 1;
    if (mat->row == 1)
    {
        len = mat->column;
    }
    else
    {
        len = mat->row;
        flag = 0;
    }
    int i;
    double sum = 0;
    for (i = 0; i < len; i++)
    {
        sum += mat->data[i * (1 - flag)][i * flag];
    }
    return sum / len;
}

/*矩阵行列式计算*/
double det_mat(const Matrix *m)
{
    unsigned int i, j, n, max_row;
    int swap_f = 0;
    double max, k, det = 1;
    if (m->column != m->row)
    {
        printf("error, matrix is not square\n");
        exit(1);
    }
    Matrix mat = copy_mat(m);
    for (i = 0; i < mat.row - 1; i++)
    {
        max = ABS(mat.data[i][i]);
        max_row = i;
        for (j = i + 1; j < mat.row; j++)
        {
            if (max < ABS(mat.data[j][i]))
            {
                max = ABS(mat.data[j][i]);
                max_row = j;
            }
        }
        if (i != max_row) //行交换
        {
            swap_f++;
            for (j = 0; j < mat.column; j++)
            {
                EXCHANGE(mat.data[i][j], mat.data[max_row][j]);
            }
        }
        for (j = i + 1; j < mat.row; j++)
        {
            k = -mat.data[j][i] / mat.data[i][i];
            for (n = 0; n < mat.column; n++)
            {
                mat.data[j][n] += mat.data[i][n] * k;
            }
        }
    }
    if (swap_f % 2 == 1)
    {
        swap_f = -1;
    }
    else
    {
        swap_f = 1;
    }
    for (i = 0; i < mat.column; i++)
    {
        det *= mat.data[i][i];
    }
    det *= swap_f;
    free_mat(&mat);
    return det;
}

/*矩阵求逆*/
Matrix inv_mat(const Matrix *m)
{
    if (det_mat(m) == 0)
    {
        printf("error, det is ZERO");
        exit(1);
    }
    Matrix mat = copy_mat(m);
    Matrix inv_mat = E_mat(m->row);
    int i, j, n, max_row;
    double max, k;
    for (i = 0; i < mat.row - 1; i++)
    {
        max = ABS(mat.data[i][i]);
        max_row = i;
        for (j = i + 1; j < mat.row; j++)
        {
            if (max < ABS(mat.data[j][i]))
            {
                max = ABS(mat.data[j][i]);
                max_row = j;
            }
        }
        if (i != max_row)
        {
            for (j = 0; j < mat.column; j++)
            {
                EXCHANGE(mat.data[i][j], mat.data[max_row][j]);
                EXCHANGE(inv_mat.data[i][j], inv_mat.data[max_row][j]);
            }
        }
        for (j = i + 1; j < mat.row; j++)
        {
            k = -mat.data[j][i] / mat.data[i][i];
            for (n = 0; n < mat.column; n++)
            {
                mat.data[j][n] += mat.data[i][n] * k;
                inv_mat.data[j][n] += inv_mat.data[i][n] * k;
            }
        }
    }
    for (i = 0; i < mat.row; i++)
    {
        k = mat.data[i][i];
        for (j = 0; j < mat.column; j++)
        {
            mat.data[i][j] /= k;
            inv_mat.data[i][j] /= k;
        }
    }
    for (i = mat.row - 1; i > 0; i = i - 1)
    {
        for (j = i - 1; j >= 0; j--)
        {
            k = -mat.data[j][i] / mat.data[i][i];
            for (n = 0; n < mat.column; n++)
            {
                mat.data[j][n] += k * mat.data[i][n];
                inv_mat.data[j][n] += k * inv_mat.data[i][n];
            }
        }
    }
    free_mat(&mat);
    return inv_mat;
}

/*LUP分解，将输入矩阵分解成L、U、P三个矩阵，需要在函数外先创建对应的矩阵并分配内存，其中L、U为空，P为单位矩阵，LU=PA，P^(-1)LU=A*/
void LUP_mat(const Matrix *m, Matrix *L, Matrix *U, Matrix *P)
{
    int i, j, k, f;
    double max;
    if (m->row != m->column)
    {
        printf("error, row is not equal to column");
        exit(1);
    }
    *U = copy_mat(m);
    for (j = 0; j < U->row - 1; j++)
    {
        max = ABS(U->data[j][j]);
        f = j;
        for (i = j; i < U->column; i++)
        {
            if (max < ABS(U->data[i][j])) //使主轴上元素是此列中最大
            {
                max = ABS(U->data[i][j]);
                f = i;
            }
        }
        for (i = 0; i < U->row; i++) //行交换
        {
            EXCHANGE(U->data[j][i], U->data[f][i]);
            EXCHANGE(P->data[j][i], P->data[f][i]);
        }
        for (i = j + 1; i < U->row; i++) //y坐标
        {
            U->data[i][j] = U->data[i][j] / U->data[j][j]; //每一列计算新的l值
            for (k = j + 1; k < U->row; k++)
            {
                U->data[i][k] = U->data[i][k] - U->data[i][j] * U->data[j][k]; //右下边每个值减去l[i][j]*u[j][k]
            }
        }
    }
    for (i = 1; i < U->row; i++)
    {
        for (j = 0; j < i; j++)
        {
            L->data[i][j] = U->data[i][j];
            U->data[i][j] = 0;
        }
    }
    for (i = 0; i < U->row; i++)
    {
        L->data[i][i] = 1;
    }
}

/*高斯消元法（利用LUP分解）解线性方程组，输入方阵地址及列向量地址*/
Matrix Gauss_mat(const Matrix *mat, const Matrix *b)
{
    if (det_mat(mat) == 0)
    {
        printf("error, det is equal to ZERO");
        exit(1);
    }
    if (mat->row != b->row)
    {
        printf("error, Length mismatch");
        exit(1);
    }
    if (b->column != 1)
    {
        printf("error, input is not vector");
        exit(1);
    }
    Matrix L = create_mat(mat->row, mat->column);
    Matrix U = create_mat(mat->row, mat->column);
    Matrix P = E_mat(mat->row);
    LUP_mat(mat, &L, &U, &P);
    Matrix Pb = mult_mat(&P, b);
    Matrix x = create_mat(b->row, 1);
    Matrix y = create_mat(b->row, 1);
    int i, j;
    double t = 0;
    for (i = 0; i < L.row; i++)
    {
        for (j = 0; j < i; j++)
        {
            t += L.data[i][j] * y.data[j][0];
        }
        y.data[i][0] = (Pb.data[i][0] - t) / L.data[i][i];
        t = 0;
    }
    for (i = U.row - 1; i >= 0; i--)
    {
        for (j = U.column - 1; j > i; j--)
        {
            t += U.data[i][j] * x.data[j][0];
        }
        x.data[i][0] = (y.data[i][0] - t) / U.data[i][i];
        t = 0;
    }
    free_mat(&L);
    free_mat(&U);
    free_mat(&P);
    free_mat(&Pb);
    free_mat(&y);
    return x;
}

/*幂法求按模最大的特征值，输入矩阵的地址以及迭代次数*/
double power_mat(const Matrix *mat, unsigned int t)
{
    Matrix x = create_mat(mat->row, 1);
    Matrix y = create_mat(mat->row, 1);
    int i, j;
    double max;
    for (i = 0; i < mat->row; i++)
    {
        x.data[i][0] = 1;
    }
    for (i = 0; i < t; i++)
    {
        y = mult_mat(mat, &x);
        max = ABS(y.data[0][0]);
        for (j = 0; j < y.row; j++)
        {
            if (max < ABS(y.data[j][0]))
            {
                max = ABS(y.data[j][0]);
            }
        }
        x = scalar_mult_mat(&y, 1 / max);
    }
    free_mat(&x);
    free_mat(&y);
    return max;
}

/*反幂法求按模最小的特征值，输入矩阵的地址以及迭代次数*/
double anti_power_mat(const Matrix *mat, unsigned int t)
{
    Matrix x = create_mat(mat->row, 1);
    Matrix y = create_mat(mat->row, 1);
    int i, j;
    double max, lambda;
    for (i = 0; i < mat->row; i++)
    {
        x.data[i][0] = 1;
    }
    for (i = 0; i < t; i++)
    {
        y = Gauss_mat(mat, &x);
        max = ABS(y.data[0][0]);
        lambda = y.data[0][0];
        for (j = 0; j < y.row; j++)
        {
            if (max < ABS(y.data[j][0]))
            {
                max = ABS(y.data[j][0]);
                lambda = y.data[j][0];
            }
        }
        x = scalar_mult_mat(&y, 1 / max);
    }
    return 1 / lambda;
}

/*创建希尔伯特矩阵*/
Matrix Hilbert_mat(const unsigned int n)
{
    Matrix mat = create_mat(n, n);
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            mat.data[i][j] = 1 / (double)(i + j + 1);
        }
    }
    return mat;
}

/*求矩阵的1范数（列模）*/
double norm_1_mat(const Matrix *mat)
{
    int i, j;
    double sum, max = 0;
    for (i = 0; i < mat->column; i++)
    {
        sum = 0;
        for (j = 0; j < mat->row; j++)
        {
            sum += ABS(mat->data[j][i]);
        }
        if (max < sum)
        {
            max = sum;
        }
    }
    return max;
}

/*求矩阵的2范数（谱模），输入矩阵的地址以及迭代次数*/
double norm_2_mat(const Matrix *mat, unsigned int t)
{
    Matrix AT = transpose_mat(mat);
    Matrix ATA = mult_mat(&AT, mat);
    double lambda = power_mat(&ATA, t);
    free_mat(&AT);
    free_mat(&ATA);
    return sqrt(lambda);
}

/*求矩阵的无穷范数（行模）*/
double norm_inf_mat(const Matrix *mat)
{
    int i, j;
    double sum, max = 0;
    for (i = 0; i < mat->row; i++)
    {
        sum = 0;
        for (j = 0; j < mat->column; j++)
        {
            sum += ABS(mat->data[i][j]);
        }
        if (max < sum)
        {
            max = sum;
        }
    }
    return max;
}

/*求矩阵的1条件数*/
double cond_1_mat(const Matrix *mat)
{
    Matrix mat_inv = inv_mat(mat);
    double cond = norm_1_mat(mat) * norm_1_mat(&mat_inv);
    free_mat(&mat_inv);
    return cond;
}

/*求矩阵的2条件数，需要迭代次数*/
double cond_2_mat(const Matrix *mat, unsigned int t)
{
    Matrix mat_T = transpose_mat(mat);
    Matrix mat_T_mat = mult_mat(&mat_T, mat);
    double lambda_1 = power_mat(&mat_T_mat, t);
    double lambda_2 = anti_power_mat(&mat_T_mat, t);
    free_mat(&mat_T);
    free_mat(&mat_T_mat);
    return sqrt(lambda_1 / lambda_2);
}

/*求矩阵的无穷条件数*/
double cond_inf_mat(const Matrix *mat)
{
    Matrix mat_inv = inv_mat(mat);
    double cond = norm_inf_mat(mat) * norm_inf_mat(&mat_inv);
    free_mat(&mat_inv);
    return cond;
}

/*求矩阵的谱半径，输入矩阵的地址以及迭代次数*/
double radius_mat(const Matrix *mat, unsigned int t)
{
    double lambda = power_mat(mat, t);
    return lambda;
}

/*Jacobi迭代法解线性方程组，输入系数矩阵地址以及列向量地址，以及迭代次数，方阵A非奇异且对角线元素不能为零，A=D+L+U（D为对角，L为下三角，U为上三角），迭代公式 x=D^(-1)[-(L+U)x+b]*/
Matrix Jacobi_mat(const Matrix *mat, const Matrix *b, unsigned int t)
{
    if (det_mat(mat) == 0)
    {
        printf("error, det is equal to ZERO");
        exit(1);
    }
    if (mat->row != b->row)
    {
        printf("error, Length mismatch");
        exit(1);
    }
    if (b->column != 1)
    {
        printf("error, input is not vector");
        exit(1);
    }
    int i, j;
    Matrix D = create_mat(mat->row, mat->column);
    Matrix L = create_mat(mat->row, mat->column);
    Matrix U = create_mat(mat->row, mat->column);
    for (i = 0; i < mat->row; i++)
    {
        for (j = 0; j < mat->column; j++)
        {
            if (i == j)
            {
                D.data[i][j] = mat->data[i][j];
            }
            if (i < j)
            {
                U.data[i][j] = mat->data[i][j];
            }
            if (i > j)
            {
                L.data[i][j] = mat->data[i][j];
            }
        }
    }
    Matrix D_inv = inv_mat(&D);
    Matrix D_inv_minus = scalar_mult_mat(&D_inv, -1);
    Matrix LU = plus_mat(&L, &U);
    Matrix J = mult_mat(&D_inv_minus, &LU);
    free_mat(&D_inv_minus);
    if (radius_mat(&J, t) >= 1)
    {
        printf("spectral radius = %8.4lf\n", radius_mat(&J, t));
        printf("error, spectral radius >= 1");
        exit(1);
    }
    Matrix f = mult_mat(&D_inv, b);
    free_mat(&D_inv);
    free_mat(&D);
    free_mat(&L);
    free_mat(&U);
    Matrix x = create_mat(mat->row, 1);
    for (i = 0; i < mat->row; i++)
    {
        x.data[i][0] = 1;
    }
    for (i = 0; i < t; i++)
    {
        x = mult_mat(&J, &x);
        x = plus_mat(&x, &f);
    }
    return x;
}

/*Gauss-Seidel迭代法解线性方程组，输入系数矩阵地址以及列向量地址，以及迭代次数，方阵A非奇异且对角线元素不能为零，A=D+L+U（D为对角，L为下三角，U为上三角），迭代公式 x=(D+L)^(-1)(-Ux+b)*/
Matrix Guass_seidel_mat(const Matrix *mat, const Matrix *b, unsigned int t)
{
    if (det_mat(mat) == 0)
    {
        printf("error, det is equal to ZERO");
        exit(1);
    }
    if (mat->row != b->row)
    {
        printf("error, Length mismatch");
        exit(1);
    }
    if (b->column != 1)
    {
        printf("error, input is not vector");
        exit(1);
    }
    int i, j;
    Matrix D = create_mat(mat->row, mat->column);
    Matrix L = create_mat(mat->row, mat->column);
    Matrix U = create_mat(mat->row, mat->column);
    for (i = 0; i < mat->row; i++)
    {
        for (j = 0; j < mat->column; j++)
        {
            if (i == j)
            {
                D.data[i][j] = mat->data[i][j];
            }
            if (i < j)
            {
                U.data[i][j] = mat->data[i][j];
            }
            if (i > j)
            {
                L.data[i][j] = mat->data[i][j];
            }
        }
    }
    Matrix DL = plus_mat(&D, &L);
    Matrix DL_inv = inv_mat(&DL);
    Matrix U_minus = scalar_mult_mat(&U, -1);
    Matrix G = mult_mat(&DL_inv, &U_minus);
    if (radius_mat(&G, t) >= 1)
    {
        printf("spectral radius = %8.4lf\n", radius_mat(&G, t));
        printf("error, spectral radius >= 1");
        exit(1);
    }
    Matrix f = mult_mat(&DL_inv, b);
    free_mat(&L);
    free_mat(&U);
    free_mat(&D);
    free_mat(&DL);
    free_mat(&DL_inv);
    free_mat(&U_minus);
    Matrix x = create_mat(mat->row, 1);
    for (i = 0; i < mat->row; i++)
    {
        x.data[i][0] = 1;
    }
    for (i = 0; i < t; i++)
    {
        x = mult_mat(&G, &x);
        x = plus_mat(&x, &f);
    }
    return x;
}

/*拉格朗日插值法进行多项式回归，输入两个列向量分别保存有序的(x_0,y_0)值对，以及需要验证的变量x*/
double Lagrange_Interpolation(const Matrix *x, const Matrix *y, double x_)
{
    if (x->column != 1 || y->column != 1)
    {
        printf("error, not vector");
        exit(1);
    }
    if (x->row != y->row)
    {
        printf("error, Length mismatch");
        exit(1);
    }
    int i, j;
    double P, L = 0;
    for (i = 0; i < x->row; i++)
    {
        P = 1;
        for (j = 0; j < x->row; j++)
        {
            if (j == i)
            {
                continue;
            }
            P *= (x_ - x->data[j][0]) / (x->data[i][0] - x->data[j][0]);
        }
        L += P * y->data[i][0];
    }
    return L;
}

/*计算拉格朗日插值法中的重心权，输入列向量形式有序的x_0*/
Matrix weight_gravity(const Matrix *x)
{
    if (x->column != 1)
    {
        printf("error, not vector");
        exit(1);
    }
    int i, j;
    Matrix w = create_mat(x->row, 1);
    for (i = 0; i < x->row; i++)
    {
        w.data[i][0] = 1;
        for (j = 0; j < x->row; j++)
        {
            if (i != j)
            {
                w.data[i][0] /= x->data[i][0] - x->data[j][0];
            }
        }
    }
    free_mat(&w);
    return w;
}

/*计算拉格朗日插值法中的l(x)，输入列向量形式有序的x_0，以及变量x*/
double Lagrange_lx(const Matrix *x, double x_)
{
    if (x->column != 1)
    {
        printf("error, not vector");
        exit(1);
    }
    int i;
    double ans = 1;
    for (i = 0; i < x->row; i++)
    {
        if (x_ == x->data[i][0])
        {
            x_ += ZERO;
        }
    }
    for (i = 0; i < x->row; i++)
    {
        ans *= x_ - x->data[i][0];
    }
    return ans;
}

/*重心拉格朗日插值法（第一型）进行多项式回归，输入两个列向量分别保存有序的(x_0,y_0)值对，以及需要验证的变量x*/
double Lagrange_Gravity_1_Interpolation(const Matrix *x, const Matrix *y, double x_)
{
    if (x->column != 1 || y->column != 1)
    {
        printf("error, not vector");
        exit(1);
    }
    if (x->row != y->row)
    {
        printf("error, Length mismatch");
        exit(1);
    }
    int i;
    double L = 0;
    Matrix w = weight_gravity(x);
    double lx = Lagrange_lx(x, x_);
    for (i = 0; i < x->row; i++)
    {
        if (x_ == x->data[i][0])
        {
            x_ += ZERO;
        }
    }
    for (i = 0; i < x->row; i++)
    {
        L += w.data[i][0] / (x_ - x->data[i][0]) * y->data[i][0];
    }
    free_mat(&w);
    return L * lx;
}

/*重心拉格朗日插值法（第二型）进行多项式回归，输入两个列向量分别保存有序的(x_0,y_0)值对，以及需要验证的变量x*/
double Lagrange_Gravity_2_Interpolation(const Matrix *x, const Matrix *y, double x_)
{
    if (x->column != 1 || y->column != 1)
    {
        printf("error, not vector");
        exit(1);
    }
    if (x->row != y->row)
    {
        printf("error, Length mismatch");
        exit(1);
    }
    Matrix w = weight_gravity(x);
    int i;
    double P1 = 0, P2 = 0;
    for (i = 0; i < x->row; i++)
    {
        if (x_ == x->data[i][0])
        {
            x_ += ZERO;
        }
    }
    for (i = 0; i < x->row; i++)
    {
        P1 += w.data[i][0] / (x_ - x->data[i][0]) * y->data[i][0];
        P2 += w.data[i][0] / (x_ - x->data[i][0]);
    }
    free_mat(&w);
    return P1 / P2;
}

/*计算牛顿插值法需要的均差*/
Matrix inequality_mat(const Matrix *x, const Matrix *y, int d)
{
    if (x->column != 1 || y->column != 1)
    {
        printf("error, not vector");
        exit(1);
    }
    int i;
    Matrix ans = create_mat(y->row - 1, 1);
    for (i = 0; i < y->row - 1; i++)
    {
        ans.data[i][0] = (y->data[i + 1][0] - y->data[i][0]) / (x->data[i + d][0] - x->data[i][0]);
    }
    return ans;
}

/*牛顿插值法进行多项式回归，输入两个列向量分别保存有序的(x_0,y_0)值对，以及需要验证的变量x*/
double Newton_Interpolation(const Matrix *x, const Matrix *y, double x_)
{
    if (x->column != 1 || y->column != 1)
    {
        printf("error, not vector");
        exit(1);
    }
    if (x->row != y->row)
    {
        printf("error, Length mismatch");
        exit(1);
    }
    Matrix is[50];
    is[0] = inequality_mat(x, y, 1);
    int i, j;
    double F = y->data[0][0], f = 0;
    for (i = 1; i < x->row; i++)
    {
        f = is[i - 1].data[0][0];
        for (j = 0; j < i; j++)
        {
            f *= (x_ - x->data[j][0]);
        }
        F += f;
        if (i < x->row - 1)
        {
            is[i] = inequality_mat(x, &is[i - 1], i + 1);
        }
    }
    for (i = 0; i < x->row - 1; i++)
    {
        free_mat(&is[i]);
    }
    return F;
}

/*计算Hermite插值法需要的l(x)，输入列向量形式有序的x，变量x_，以及参数i*/
double Hermite_lx(const Matrix *x, double x_, int i)
{
    if (x->column != 1)
    {
        printf("error, not vector");
        exit(1);
    }
    if (i < 0 || i > x->row)
    {
        printf("error, overstep");
        exit(1);
    }
    int j;
    double l = 1;
    for (j = 0; j < x->row; j++)
    {
        if (j == i)
        {
            continue;
        }
        l *= (x_ - x->data[j][0]) / (x->data[i][0] - x->data[j][0]);
    }
    return l;
}

/*计算Hermite插值法需要的l'(x)，输入列向量形式有序的x，以及参数i*/
double Hermite_lx_(const Matrix *x, int i)
{
    if (x->column != 1)
    {
        printf("error, not vector");
        exit(1);
    }
    if (i < 0 || i > x->row)
    {
        printf("error, overstep");
        exit(1);
    }
    int j;
    double l = 1;
    for (j = 0; j < x->row; j++)
    {
        if (j == i)
        {
            continue;
        }
        l /= (x->data[i][0] - x->data[j][0]);
    }
    return l;
}

/*二次Hermite插值法，输入三个列向量x,y,y'，以及需要验证的变量x_*/
double Hermite_2_Interpolation(const Matrix *x, const Matrix *y, const Matrix *y_, double x_)
{
    if (x->column != 1 || y->column != 1 || y_->column != 1)
    {
        printf("error, not vector");
        exit(1);
    }
    if (x->row != y->row || x->row != y_->row)
    {
        printf("error, Length mismatch");
        exit(1);
    }
    int i;
    double h = 0;
    for (i = 0; i < x->row; i++)
    {
        h += (1 - 2 * (x_ - x->data[i][0]) * Hermite_lx_(x, i)) * Hermite_lx(x, x_, i) * Hermite_lx(x, x_, i) * y->data[i][0];
        h += (x_ - x->data[i][0]) * Hermite_lx(x, x_, i) * Hermite_lx(x, x_, i) * y_->data[i][0];
    }
    return h;
}

/*计算步长，输入列向量，返回长度减一的向量存储步长*/
Matrix step_length(const Matrix *x)
{
    if (x->column != 1)
    {
        printf("error, not vector");
        exit(1);
    }
    int i;
    Matrix h = create_mat(x->row - 1, 1);
    for (i = 0; i < x->row - 1; i++)
    {
        h.data[i][0] = x->data[i + 1][0] - x->data[i][0];
    }
    return h;
}

/*三次样条插值（自由边界），输入两个列向量x,y，生成一个存储了四组系数的矩阵*/
Matrix spline_natural(const Matrix *x, const Matrix *y)
{
    if (x->column != 1 || y->column != 1)
    {
        printf("error, not vector");
        exit(1);
    }
    if (x->row != y->row)
    {
        printf("error, Length mismatch");
        exit(1);
    }
    Matrix h = step_length(x);
    Matrix M = create_mat(x->row, x->row);
    int i, j;
    M.data[0][0] = 1;
    M.data[x->row - 1][x->row - 1] = 1;
    for (i = 1; i < x->row - 1; i++)
    {
        M.data[i][i - 1] = h.data[i - 1][0];
        M.data[i][i] = 2 * (h.data[i - 1][0] + h.data[i][0]);
        M.data[i][i + 1] = h.data[i][0];
    }
    Matrix z = create_mat(x->row, 1);
    for (i = 1; i < x->row - 1; i++)
    {
        z.data[i][0] = 6 * (y->data[i + 1][0] - y->data[i][0]) / h.data[i][0] - (y->data[i][0] - y->data[i - 1][0]) / h.data[i - 1][0];
    }
    Matrix m = Gauss_mat(&M, &z);
    Matrix a = create_mat(x->row - 1, 4);
    for (i = 0; i < x->row - 1; i++)
    {
        a.data[i][0] = y->data[i][0];
        a.data[i][1] = (y->data[i + 1][0] - y->data[i][0]) / h.data[i][0] - h.data[i][0] * (m.data[i + 1][0] + 2 * m.data[i][0]) / 6;
        a.data[i][2] = m.data[i][0] / 2;
        a.data[i][3] = (m.data[i + 1][0] - m.data[i][0]) / (6 * h.data[i][0]);
    }
    free_mat(&h);
    free_mat(&M);
    free_mat(&z);
    free_mat(&m);
    return a;
}

/*三次样条插值（固定边界），输入两个列向量x,y，给定两端点处微分值A、B，生成一个存储了四组系数的矩阵*/
Matrix spline_clamped(const Matrix *x, const Matrix *y, double A, double B)
{
    if (x->column != 1 || y->column != 1)
    {
        printf("error, not vector");
        exit(1);
    }
    if (x->row != y->row)
    {
        printf("error, Length mismatch");
        exit(1);
    }
    Matrix h = step_length(x);
    Matrix M = create_mat(x->row, x->row);
    int i, j;
    M.data[0][0] = 2 * h.data[0][0];
    M.data[0][1] = h.data[0][0];
    M.data[x->row - 1][x->row - 1] = 2 * h.data[x->row - 2][0];
    M.data[x->row - 1][x->row - 2] = h.data[x->row - 2][0];
    for (i = 1; i < x->row - 1; i++)
    {
        M.data[i][i - 1] = h.data[i - 1][0];
        M.data[i][i] = 2 * (h.data[i - 1][0] + h.data[i][0]);
        M.data[i][i + 1] = h.data[i][0];
    }
    Matrix z = create_mat(x->row, 1);
    z.data[0][0] = 6 * ((y->data[1][0] - y->data[0][0]) / h.data[0][0] - A);
    z.data[x->row - 1][0] = 6 * (B - (y->data[x->row - 1][0] - y->data[x->row - 2][0]) / h.data[x->row - 2][0]);
    for (i = 1; i < x->row - 1; i++)
    {
        z.data[i][0] = 6 * (y->data[i + 1][0] - y->data[i][0]) / h.data[i][0] - (y->data[i][0] - y->data[i - 1][0]) / h.data[i - 1][0];
    }
    Matrix m = Gauss_mat(&M, &z);
    Matrix a = create_mat(x->row - 1, 4);
    for (i = 0; i < x->row - 1; i++)
    {
        a.data[i][0] = y->data[i][0];
        a.data[i][1] = (y->data[i + 1][0] - y->data[i][0]) / h.data[i][0] - h.data[i][0] * (m.data[i + 1][0] + 2 * m.data[i][0]) / 6;
        a.data[i][2] = m.data[i][0] / 2;
        a.data[i][3] = (m.data[i + 1][0] - m.data[i][0]) / (6 * h.data[i][0]);
    }
    free_mat(&h);
    free_mat(&M);
    free_mat(&z);
    free_mat(&m);
    return a;
}

/*三次样条插值（非节点边界），输入两个列向量x,y，生成一个存储了四组系数的矩阵*/
Matrix spline_NAK(const Matrix *x, const Matrix *y)
{
    if (x->column != 1 || y->column != 1)
    {
        printf("error, not vector");
        exit(1);
    }
    if (x->row != y->row)
    {
        printf("error, Length mismatch");
        exit(1);
    }
    Matrix h = step_length(x);
    Matrix M = create_mat(x->row, x->row);
    int i, j;
    M.data[0][0] = -h.data[1][0];
    M.data[0][1] = h.data[0][0] + h.data[1][0];
    M.data[0][2] = h.data[0][0];
    M.data[x->row - 1][x->row - 1] = -h.data[x->row - 3][0];
    M.data[x->row - 1][x->row - 2] = h.data[x->row - 2][0] + h.data[x->row - 3][0];
    M.data[x->row - 1][x->row - 3] = -h.data[x->row - 2][0];
    for (i = 1; i < x->row - 1; i++)
    {
        M.data[i][i - 1] = h.data[i - 1][0];
        M.data[i][i] = 2 * (h.data[i - 1][0] + h.data[i][0]);
        M.data[i][i + 1] = h.data[i][0];
    }
    Matrix z = create_mat(x->row, 1);
    for (i = 1; i < x->row - 1; i++)
    {
        z.data[i][0] = 6 * (y->data[i + 1][0] - y->data[i][0]) / h.data[i][0] - (y->data[i][0] - y->data[i - 1][0]) / h.data[i - 1][0];
    }
    Matrix m = Gauss_mat(&M, &z);
    Matrix a = create_mat(x->row - 1, 4);
    for (i = 0; i < x->row - 1; i++)
    {
        a.data[i][0] = y->data[i][0];
        a.data[i][1] = (y->data[i + 1][0] - y->data[i][0]) / h.data[i][0] - h.data[i][0] * (m.data[i + 1][0] + 2 * m.data[i][0]) / 6;
        a.data[i][2] = m.data[i][0] / 2;
        a.data[i][3] = (m.data[i + 1][0] - m.data[i][0]) / (6 * h.data[i][0]);
    }
    free_mat(&h);
    free_mat(&M);
    free_mat(&z);
    free_mat(&m);
    return a;
}

/*利用输入的样条插值矩阵进行计算，输入样例点向量和变量x_，给出拟合得到的函数值*/
double spline_out(const Matrix *m, const Matrix *x, double x_)
{
    int i;
    for (i = 0; i < m->row; i++)
    {
        if (x->data[i][0] <= x_ && x->data[i + 1][0] >= x_)
        {
            return m->data[i][0] + m->data[i][1] * (x_ - x->data[i][0]) + m->data[i][2] * (x_ - x->data[i][0]) * (x_ - x->data[i][0]) + m->data[i][3] * (x_ - x->data[i][0]) * (x_ - x->data[i][0]) * (x_ - x->data[i][0]);
        }
    }
    printf("out of range");
    exit(1);
}

/*最小二乘法进行多元线性回归，输出一组列向量存储系数*/
Matrix least_square(const Matrix *x_, const Matrix *y)
{
    Matrix ones = ones_mat(x_->row, 1);
    Matrix x = joint_mat(&ones, x_, 2);
    Matrix x_T = transpose_mat(&x);
    Matrix x_T_x = mult_mat(&x_T, &x);
    Matrix x_T_x_inv = inv_mat(&x_T_x);
    Matrix x_T_y = mult_mat(&x_T, y);
    Matrix w = mult_mat(&x_T_x_inv, &x_T_y);
    free_mat(&ones);
    free_mat(&x);
    free_mat(&x_T);
    free_mat(&x_T_x);
    free_mat(&x_T_x_inv);
    free_mat(&x_T_y);
    return w;
}

/*根据给出的系数向量，计算线性函数值*/
double linear_value(const Matrix *w, const Matrix *x)
{
    Matrix w_T = transpose_mat(w);
    Matrix one = ones_mat(1, 1);
    Matrix x_ = joint_mat(&one, x, 1);
    Matrix v = mult_mat(&w_T, &x_);
    free_mat(&w_T);
    free_mat(&one);
    free_mat(&x_);
    return v.data[0][0];
}

/*计算误差平方和*/
double SSE(const Matrix *x_, const Matrix *y, const Matrix *w)
{
    Matrix ones = ones_mat(x_->row, 1);
    Matrix x = joint_mat(&ones, x_, 2);
    int i, j;
    double sum = 0, y_ = 0;
    for (i = 0; i < x_->row; i++)
    {
        for (j = 0; j < x.column; j++)
        {
            y_ += w->data[j][0] * x.data[i][j];
        }
        sum += (y->data[i][0] - y_) * (y->data[i][0] - y_);
        y_ = 0;
    }
    return sum;
}

/*计算总修正平均值*/
double SST(const Matrix *x_, const Matrix *y, const Matrix *w)
{
    Matrix ones = ones_mat(x_->row, 1);
    Matrix x = joint_mat(&ones, x_, 2);
    int i, j;
    double sum = 0, y_ = 0;
    for (i = 0; i < x_->row; i++)
    {
        for (j = 0; j < x.column; j++)
        {
            y_ += w->data[j][0] * x.data[i][j];
        }
    }
    y_ = y_ / y->row;
    for (i = 0; i < x_->row; i++)
    {
        sum += (y->data[i][0] - y_) * (y->data[i][0] - y_);
    }
    return sum;
}

/*计算决定系数R^2*/
double R_square(const Matrix *x_, const Matrix *y, const Matrix *w)
{
    return 1 - SSE(x_, y, w) / SST(x_, y, w);
}

/*对于给定的两个变量相同的级数系数向量，给出其乘积的系数向量*/
Matrix polynomial_mult(const Matrix *a, const Matrix *b)
{
    Matrix ans = create_mat(a->row + b->row - 1, 1);
    int i, j;
    for (i = 0; i < a->row; i++)
    {
        for (j = 0; j < b->row; j++)
        {
            ans.data[i + j][0] += a->data[i][0] * b->data[j][0];
        }
    }
    return ans;
}

/*对指定的变量x，计算用系数向量表示的级数的值*/
double polynomial_value(const Matrix *p, double x)
{
    if (p->column != 1)
    {
        printf("error, column is not equal to 1");
        exit(1);
    }
    int i;
    double ans = 0;
    for (i = 0; i < p->row; i++)
    {
        ans += p->data[i][0] * pow(x, i);
    }
    return ans;
}

/*正交多项式回归法，输入数据向量x和y，以及权重向量w，要求三个向量长度一样，误差上限TOL，以及最大次数max*/
Matrix orthogonal_polynomial_approximation(const Matrix *x, const Matrix *y, const Matrix *w, double TOL, int max)
{
    if (x->column != 1 || y->column != 1 || w->column != 1)
    {
        printf("error, not vector");
        exit(1);
    }
    if (x->row != y->row || x->row != w->row)
    {
        printf("error, Length mismatch");
        exit(1);
    }
    Matrix phi[10];
    phi[0] = ones_mat(1, 1); //phi_0(x)=1
    //show_mat(&phi[0], "phi[0]");
    int i, j, len = x->row;
    double a, B, C, err, n = 0, m = 0;
    Matrix P = create_mat(10, 1);
    Matrix D, E, F;
    for (i = 0; i < len; i++)
    {
        n += polynomial_value(&phi[0], x->data[i][0]) * y->data[i][0] * w->data[i][0]; //(phi_0,y)
        m += pow(polynomial_value(&phi[0], x->data[i][0]), 2) * w->data[i][0];         //(phi_0,phi_0)
    }
    a = n / m; //a_0=(phi_0,y)/(phi_0,phi_0)
    D = scalar_mult_mat(&phi[0], a);
    //show_mat(&D, "D");
    P = plus_mat_2(&P, &D); //P(x)=a_0phi_0(x)
    //show_mat(&P, "P");
    n = 0;
    m = 0;
    for (i = 0; i < len; i++)
    {
        n += y->data[i][0] * y->data[i][0] * w->data[i][0];                            //(y,y)
        m += polynomial_value(&phi[0], x->data[i][0]) * y->data[i][0] * w->data[i][0]; //(phi_0,y)
    }
    err = n - a * m; //err=(y,y)-a_0(phi_0,y)
    //printf("%lf\n", err);
    n = 0;
    m = 0;
    D = create_mat(phi[0].row + 1, 1);
    //show_mat(&D, "D");
    for (i = phi[0].row; i > 0; i--)
    {
        D.data[i][0] = phi[0].data[i - 1][0];
    }
    D.data[0][0] = 0; //D=x*phi_0
    //show_mat(&D, "D");
    for (i = 0; i < len; i++)
    {
        n += polynomial_value(&D, x->data[i][0]) * polynomial_value(&phi[0], x->data[i][0]) * w->data[i][0]; //(x*phi_0,y)
        m += pow(polynomial_value(&phi[0], x->data[i][0]), 2) * w->data[i][0];                               //(phi_0,phi_0)
    }
    //printf("n=%lf\n", n);
    //printf("m=%lf\n", m);
    B = n / m;
    //printf("B=%lf\n", B);
    D = create_mat(2, 1);
    D.data[0][0] = -B;
    D.data[1][0] = 1;                      //D=x-B_1
    phi[1] = polynomial_mult(&D, &phi[0]); //phi_1=x-B_1
    n = 0;
    m = 0;
    for (i = 0; i < len; i++)
    {
        n += polynomial_value(&phi[1], x->data[i][0]) * y->data[i][0] * w->data[i][0]; //(phi_1,y)
        m += pow(polynomial_value(&phi[1], x->data[i][0]), 2) * w->data[i][0];         //(phi_1,phi_1)
    }
    a = n / m; //a_1=(phi_1,y)/(phi_1,phi_1)
    //printf("a=%lf\n", a);
    D = scalar_mult_mat(&phi[1], a); //D=a_1*phi_1
    P = plus_mat_2(&P, &D);          //P+=a_1*phi_1
    //show_mat(&P, "P");
    for (i = 0; i < len; i++)
    {
        err -= a * polynomial_value(&phi[1], x->data[i][0]) * y->data[i][0] * w->data[i][0]; //err-=a_1*(phi_1,y)
    }
    int k = 1;
    while (k < max && ABS(err) >= TOL)
    {
        k++;
        D = create_mat(phi[k - 1].row + 1, 1);
        for (i = phi[k - 1].row; i > 0; i--)
        {
            D.data[i][0] = phi[k - 1].data[i - 1][0];
        }
        D.data[0][0] = 0; //D=x*phi_(k-1)
        //show_mat(&D, "D");
        n = 0;
        m = 0;
        for (i = 0; i < len; i++)
        {
            n += polynomial_value(&D, x->data[i][0]) * polynomial_value(&phi[k - 1], x->data[i][0]) * w->data[i][0]; //(x*phi_(k-1),phi_(k-1))
            m += pow(polynomial_value(&phi[k - 1], x->data[i][0]), 2) * w->data[i][0];                               //(phi_(k-1), phi_(k-1))
        }
        //printf("n=%lf\n", n);
        //printf("m=%lf\n", m);
        B = n / m;
        //printf("B=%lf\n", B);
        n = 0;
        m = 0;
        for (i = 0; i < len; i++)
        {
            n += polynomial_value(&D, x->data[i][0]) * polynomial_value(&phi[k - 2], x->data[i][0]) * w->data[i][0]; //(x*phi_(k-1),phi_(k-2))
            m += pow(polynomial_value(&phi[k - 2], x->data[i][0]), 2) * w->data[i][0];                               //(phi_(k-2), phi_(k-2))
        }
        //printf("n=%lf\n", n);
        //printf("m=%lf\n", m);
        C = n / m;
        //printf("C=%lf\n", C);
        D = create_mat(2, 1);
        D.data[0][0] = -B;
        D.data[1][0] = 1;                     //D=x-B_k
        E = polynomial_mult(&D, &phi[k - 1]); //E=(x-B_1)phi_(k-1)
        F = scalar_mult_mat(&phi[k - 2], C);  //F=C*phi_(k-2)
        phi[k] = minus_mat_2(&E, &F);
        n = 0;
        m = 0;
        for (i = 0; i < len; i++)
        {
            n += polynomial_value(&phi[k], x->data[i][0]) * y->data[i][0] * w->data[i][0]; //(phi_k,y)
            m += pow(polynomial_value(&phi[k], x->data[i][0]), 2) * w->data[i][0];         //(phi_k,phi_k)
        }
        a = n / m; //a_k=(phi_k,y)/(phi_k,phi_k)
        // printf("a=%lf\n", a);
        D = scalar_mult_mat(&phi[k], a); //D=a_k*phi_k
        P = plus_mat_2(&P, &D);          //P+=a_k*phi_k
        for (i = 0; i < len; i++)
        {
            err -= a * polynomial_value(&phi[k], x->data[i][0]) * y->data[i][0] * w->data[i][0]; //err-=a_k*(phi_k,y)
        }
    }
    Matrix new_P;
    for (i = 9; i >= 0; i--)
    {
        if (P.data[i][0] != 0)
        {
            new_P = interception_mat(&P, 0, 0, 0, i);
            break;
        }
    }
    return new_P;
}

/*用三点中心差分计算数值微分，输入函数指针（一元函数，参数和返回值为double类型），以及变量x*/
double diffrent(double (*fun)(double), double x)
{
    int i, j;
    double h = 0.1, init;
    double ans = ((*fun)(x + h) - (*fun)(x - h)) / (2 * h);
    do
    {
        init = ans;
        h /= 10;
        ans = ((*fun)(x + h) - (*fun)(x - h)) / (2 * h);
    } while (ABS(ans - init) > 1e-6);
    return ans;
}

/*用三点中心差分计算二阶数值微分，输入函数指针（一元函数，参数和返回值为double类型），以及变量x*/
double diffrent_2(double (*fun)(double), double x)
{
    int i, j;
    double h = 0.1, init;
    double ans = ((*fun)(x + h) + (*fun)(x - h) - 2 * (*fun)(x)) / (h * h);
    do
    {
        init = ans;
        h /= 10;
        ans = ((*fun)(x + h) + (*fun)(x - h) - 2 * (*fun)(x)) / (h * h);
    } while (ABS(ans - init) > 1e-6);
    return ans;
}

/*用梯形法则计算数值积分，输入函数指针（一元函数，参数和返回值为double类型），以及积分上下限b、a(b>a)*/
double trapezoidal_integration(double (*fun)(double), double a, double b)
{
    if (a > b)
    {
        printf("error, a > b");
        exit(1);
    }
    return ((*fun)(a) + (*fun)(b)) * (b - a) / 2;
}

/*用辛普森法则计算数值积分，输入函数指针（一元函数，参数和返回值为double类型），以及积分上下限b、a(b>a)*/
double simpson_integation(double (*fun)(double), double a, double b)
{
    if (a > b)
    {
        printf("error, a > b");
        exit(1);
    }
    return ((*fun)(a) + (*fun)(b) + 4 * (*fun)(a + (b - a) / 2)) * (b - a) / 6;
}

/*用复合梯形法则计算数值积分，输入函数指针（一元函数，参数和返回值为double类型），以及积分上下限b、a(b>a)，区间数量m*/
double composite_trapezoidal_integration(double (*fun)(double), double a, double b, int m)
{
    if (a > b)
    {
        printf("error, a > b");
        exit(1);
    }
    double h = (b - a) / m, ans = 0, t = a;
    int i;
    ans += (*fun)(a) + (*fun)(b);
    for (i = 1; i < m; i++)
    {
        t += h;
        ans += 2 * (*fun)(t);
    }
    return ans * h / 2;
}

/*用复合辛普森法则计算数值积分，输入函数指针（一元函数，参数和返回值为double类型），以及积分上下限b、a(b>a)，区间数量m*/
double composite_simpson_integration(double (*fun)(double), double a, double b, int m)
{
    if (a > b)
    {
        printf("error, a > b");
        exit(1);
    }
    double h = (b - a) / (2 * m), ans = 0;
    int i;
    ans += (*fun)(a) + (*fun)(b);
    for (i = 1; i < m + 1; i++)
    {
        ans += 4 * (*fun)(a + (2 * i - 1) * h);
    }
    for (i = 1; i < m; i++)
    {
        ans += 2 * (*fun)(a + (2 * i) * h);
    }
    return ans * h / 3;
}

/*用复合中点法则计算数值积分，输入函数指针（一元函数，参数和返回值为double类型），以及积分上下限b、a(b>a)*/
double composite_midpoint_integration(double (*fun)(double), double a, double b)
{
    if (a > b)
    {
        printf("error, a > b");
        exit(1);
    }
    double h = (b - a) / 4, ans = 0;
    return 4 * h / 3 * (2 * (*fun)(a + h) - (*fun)(a + 2 * h) + 2 * (*fun)(a + 3 * h));
}

/*龙贝格积分，输入函数指针（一元函数，参数和返回值为double类型），以及积分上下限b、a(b>a)，深度m*/
double romberg_integration(double (*fun)(double), double a, double b, int m)
{
    if (a > b)
    {
        printf("error, a > b");
        exit(1);
    }
    Matrix R = create_mat(m, m);
    R.data[0][0] = (b - a) / 2 * ((*fun)(a) + (*fun)(b));
    int i, j;
    double t, h;
    for (j = 1; j < m; j++)
    {
        R.data[j][0] = R.data[j - 1][0] / 2;
        t = 0;
        h = (b - a) / pow(2, j);
        for (i = 1; i <= pow(2, j - 1); i++)
        {
            t += (*fun)(a + (2 * i - 1) * h);
        }
        R.data[j][0] += t * h;
    }
    for (i = 1; i < m; i++)
    {
        for (j = i; j < m; j++)
        {
            R.data[j][i] = (pow(4, i) * R.data[j][i - 1] - R.data[j - 1][i - 1]) / (pow(4, i) - 1); // R[j][k]=(4^(k-1)R[j][k-1]-R[j-1][k-1])/(4^(k-1)-1)
        }
    }
    return R.data[m - 1][m - 1];
}

/*自适应积分（利用辛普森法则），输入函数指针（一元函数，参数和返回值为double类型），以及积分上下限b、a(b>a)，误差限TOL*/
//实质就是比较一下一块的积分和切成两半的积分之间的差距，如果比较大就分割成小块，如果差距不大就按整块算，像这样不断二分的分割成小块，利用递归容易实现。
double adaptive_integration(double (*fun)(double), double a, double b, double TOL)
{
    if (a > b)
    {
        printf("error, a > b");
        exit(1);
    }
    double S_0 = simpson_integation((*fun), a, b);
    double c = a + (b - a) / 2;
    double S_1 = simpson_integation((*fun), a, c), S_2 = simpson_integation((*fun), c, b);
    // printf("%lf\n", S_0);
    if (ABS(S_0 - S_1 - S_2) < 10 * TOL)
    {
        return S_1 + S_2 - (S_1 + S_2 - S_0) / 10;
    }
    return adaptive_integration((*fun), a, c, TOL / 2) + adaptive_integration((*fun), c, b, TOL / 2);
}

/*高斯积分，输入函数指针（一元函数，参数和返回值为double类型），以及积分上下限b、a(b>a)，高斯积分点n（最大为4）*/
double gauss_integration(double (*fun)(double), double a, double b, int n)
{
    if (a > b)
    {
        printf("error, a > b");
        exit(1);
    }
    Matrix gauss_int_table = create_mat(10, 3);
    double data[] = {
        1, 0, 2,
        2, -0.57735026918963, 1,
        2, 0.57735026918963, 1,
        3, -0.77459666924148, 0.55555555555555,
        3, 0, 0.88888888888888,
        3, 0.77459666924148, 0.55555555555555,
        4, -0.86113631159405, 0.34785484513745,
        4, -0.33998104358486, 0.65214515486255,
        4, 0.33998104358486, 0.65214515486255,
        4, 0.86113631159405, 0.34785484513745};
    set_mat(&gauss_int_table, data);
    double m[4], sum = 0;
    int i, t = 0;
    for (i = 0; i < n; i++)
    {
        t += i;
    }
    for (i = 0; i < n; i++)
    {
        m[i] = 0.5 * (a + b) + 0.5 * (b - a) * gauss_int_table.data[t + i][1];
        sum += 0.5 * (b - a) * gauss_int_table.data[t + i][2] * (*fun)(m[i]);
    }
    return sum;
}

double f(double x)
{
    return x * x;
}

int main()
{
    /*Matrix x = create_mat(4, 1);
    double data_x[] = {
        1,
        2,
        3,
        4};
    set_mat(&x, data_x);
    Matrix y = create_mat(4, 1);
    double data_y[] = {
        4,
        10,
        18,
        26};
    set_mat(&y, data_y);
    Matrix w = create_mat(4, 1);
    double data_w[] = {
        1,
        1,
        1,
        1};
    set_mat(&w, data_w);
    Matrix x_ = create_mat(2, 1);
    double data_x_[] = {
        2, 1, 3,
        0, 1, 3};
    set_mat(&x_, data_x_);
    Matrix m = least_square(&x, &y);
    show_mat(&m, "m");
    printf("%lf\n", SSE(&x, &y, &m));
    printf("%lf\n", SST(&x, &y, &m));
    printf("%lf\n", R_square(&x, &y, &m));
    Matrix ans = minus_mat_2(&x, &x_);
    show_mat(&ans, "ans");
    printf("%lf\n", Newton_Interpolation(&x, &y, 4));
    ans = orthogonal_polynomial_approximation(&x, &y, &w, 0.0001, 2);
    show_mat(&ans, "ans");*/
    printf("%lf\n", gauss_integration(f, 2, 4, 3));
    /*
    Matrix mat = create_mat(3, 3);
    double data[] = {
        1, 1, 0.5,
        1, 1, 0.25,
        0.5, 0.25, 2};
    set_mat(&mat, data);
    show_mat(&mat, "mat");
    Matrix inverse_mat = inv_mat(&mat);
    show_mat(&inverse_mat, "inverse_mat");
    Matrix new_mat = mult_mat(&mat, &inverse_mat);
    show_mat(&new_mat, "new_mat");
    new_mat = plus_mat(&mat, &inverse_mat);
    show_mat(&new_mat, "new_mat");
    Matrix L = create_mat(3, 3), U = create_mat(3, 3), P = E_mat(3);
    LUP_mat(&mat, &L, &U, &P);
    show_mat(&L, "L");
    show_mat(&U, "U");
    show_mat(&P, "P");
    printf("eigenvalue_min=%lf\n", anti_power_mat(&mat, 20));
    double data_a[] = {
        7, 1, 2,
        1, 8, 2,
        2, 2, 9};
    double data_b[] = {
        10,
        8,
        6};
    Matrix A = create_mat(3, 3);
    Matrix b = create_mat(3, 1);
    set_mat(&A, data_a);
    set_mat(&b, data_b);
    Matrix z = Jacobi_mat(&A, &b, 50);
    show_mat(&z, "z");
    Matrix H = Hilbert_mat(3);
    show_mat(&H, "H");
    Matrix C = create_mat(2, 2);
    double data_c[] = {
        1, 2,
        1.00001, 2};
    set_mat(&C, data_c);
    Matrix C_inv = inv_mat(&C);
    show_mat(&C_inv, "C_inv");
    */
}