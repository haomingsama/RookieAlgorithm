'''

分礼物问题，把n个礼物分给k个小朋友。输出一共有多少种分法，以及每种具体的分法

方法： 用动态规划的方法来分。构建二维矩阵来记录每种n和k配对时的情况

'''


##原始法
import itertools


def brute_split(n,k):
    '''
    原始法可以直接穷尽所有的可能，利用条件搜索出符合条件的方案
    :param n: 礼物的数量
    :param k: 小朋友的数量
    :return: len(plan) 所有符合条件的分配方案
    '''
    plan = [i for i in itertools.product(range(n+1),repeat = k) if sum(i) ==n]

    return len(plan)




##动态规划法

import numpy as np
from scipy.special import comb, perm

def create_matrix(n,k):
    '''

    :param n: 礼物的数量
    :param k: 小朋友的数量
    :return: 初始化的二维矩阵
    '''

    matrix = np.zeros([k+1,n+1],dtype = int)
    matrix[1] = 1  # 设置第一行
    #设置第0列
    for i in range(1,k+1):
        matrix[i][0] = 1
    #设置第1列
    for i in range(1,k+1):
        matrix[i][1] = i
    return matrix

# print(create_matrix(5,3))



def split(n,k,matrix,Tmatrix):
    '''
    利用动态规划更新法，求出分糖果的方案总数
    :param n: 礼物的数量
    :param k: 小朋友的数量
    :param matrix: 初始化的二维矩阵，用于保存最终方案的数量
    :param Tmatrix: 初始化的二维矩阵，用于保存具体的方案

    :return: 将n个糖果分给k个小朋友的方案总数

    用法：
    split(3,2,matrix) ==> 4  （将三个糖果分给两个小朋友有4种分法）
    '''
    for i in range(2,k+1):
        for j in range(2,n+1):
            new_tuple = []        # 每次循环一格，都要把方案列表清空
            if j>=i:              #礼物的数量>小朋友数量
                for a in range(i):
                    if a ==0:
                        matrix[i][j] += comb(i,a)*matrix[i][j-i]
                        new_tuple_list = get_tuple(j, i, a, Tmatrix)        #对于这个分支，获取所有的方案
                        new_tuple += new_tuple_list                         #将方案放进new_tuple中
                    else:
                        matrix[i][j] += comb(i,a)*matrix[i-a][j-(i-a)]
                        new_tuple_list = get_tuple(j, i, a, Tmatrix)         #对于这个分支，获取所有的方案
                        new_tuple += new_tuple_list                          #将方案放进new_tuple中
                Tmatrix[i][j] = new_tuple                                    #更新Tmatrix的方案单元格



            else:                 #小朋友数量>礼物的数量
                for a in range(i-j,i):
                    matrix[i][j] += comb(i, a) * matrix[i - a][j - (i - a)]
                    new_tuple_list = get_tuple(j, i, a, Tmatrix)              #对于这个分支，获取所有的方案
                    new_tuple += new_tuple_list                               #将方案放进new_tuple中
                Tmatrix[i][j] = new_tuple                                      #更新Tmatrix的方案单元格

    return matrix[k][n], Tmatrix[k][n]





def generate_tuple(k,s=1):
    '''
    为Tmatrix 的第0列和第1列生成 生成对应的tuple
    :param k: 小朋友的数量
    :param s: 当s = 0 时，生成的是第0列的单元格里面的tuple。
              当s = 1 时， 生成的是第1列的单元格里面的tuple。
    '''
    if s ==0:
        a =[tuple([0 for i in range(k)])]   # 第0列都是关于0的tuple
        return a
    #第1列的tuple可以用itertools的工具来遍历生成
    a = [i for i in itertools.product(range(2),repeat=k) if sum(i)==1]  #注意，只筛选出只有1个'1'在场的tuple
    return a




def init_Tmatrix(n,k):
    '''
    初始化T 矩阵，存放每种情况的分配方案
    :param n: 礼物的数量
    :param k: 小朋友的数量
    :return: 初始化的二维矩阵
    '''
    #用此种方式可以创建列表中的列表。（而且列表里的值不会一起变化）
    a = [[1 for i in range(n+1)] for x in range(k+1)]

    #初始化第一行
    for i in range(1,n+1):
        a[1][i] = [(i,)]

    #初始化第0列以及第1列
    for j in range(1,k+1):
        a[j][1] = generate_tuple(j)
        a[j][0] = generate_tuple(j,0)

    return a

def match(i,index,k):
    '''
    因为子方案和基方案的size不一样，需要将子方案进行补零。
    具体的操作是先生成全为0，size和k一样的列表。
    对于下标位置不在index的下标，按子方案中的数的顺序进行填充。

    :param i: 给定的子方案，需要补零
    :param index: 给定的0的下标
    :param k: 一共有多少个小朋友，决定了tuple的size
    :return: 补完0的子方案
    '''
    new_i = [0 for a in range(k)]    #生成全为0的列表
    count = 0
    for num,x in enumerate(new_i):   #对于不为0的下标
        if num not in index:
            new_i[num] =i[count]    #按子方案的顺序进行填充
            count+=1
    return new_i




def get_tuple(n, k, p, Tmatrix):
    '''
    在给定n, k, 以及p的情况下，输出所有的具体方案

    :param n: 给定的礼物数量
    :param k: 给定的小朋友数量
    :param p: 没有拿到礼物的小朋友数量
        p = 0 ==> 所有小朋友都有礼物
        p = 1 ==> 有1个小朋友没有礼物

    :param Tmatrix: 由init_Tmatrix(n,k)初始化好的Tmatirx

    :return: 在给定n, k, 以及p的情况下，输出所有的具体方案。

    '''
    case = int(comb(k, p))   #在当前p下，所有可能的基 方案
    if p == 0:              #如果p = 0, 基方案只有1种，就是所有小朋友都先拿到了1个礼物
        base_tuple = tuple([1 for x in range(k)])
        new_tuple = [tuple(np.array(base_tuple) + np.array(i)) for i in Tmatrix[k][n - k]] #这种情况，子分配任务下的方案size肯定和基方案是一样的，所以不用补零

    else:                   #p不是0的时候，基方案就会有多种，
        new_tuple = []
        index = list(itertools.combinations(range(k), r=p))   #用这个工具生成0的位置下标。 比如（1，4）代表第1和第4下标位的值为0
        for i in range(case):   #遍历每种基方案
            base_tuple = [1 for x in range(k)]  #首先初始化基方案为所有数值都为1的tuple
            for zero_index in index[i]:         #在相应的下标，将1修改成0，代表这个位置的小朋友没有礼物
                base_tuple[zero_index] = 0
            base_tuple = tuple(base_tuple)      #因为tuple是不可变的，所以要先用list代替，最后转成tuple

            for a in Tmatrix[k - p][n - k + p]: #在每种基方案下，遍历我们的子方案。
                match_tuple = match(a, index[i], k)   #子方案的tuple，size和我们的基方案不一样，需要补零
                new_tuple.append(tuple(np.array(base_tuple) + np.array(match_tuple))) #将基方案和子方案相加。因为tuple不能broadcast，需要先转换成np array

    return new_tuple




#测试代码正确性

n= 10
k =5

matrix = create_matrix(n,k)
Tmatrix = init_Tmatrix(n,k)
plan_num, plans = split(n,k,matrix,Tmatrix)

print(f'动态规划法：有{n}个礼物，要分给{k} 个小朋友，分法总共有{plan_num}种')
print('原始法:',brute_split(n,k))
# print('动态规划法：',split(n,k,matrix))

print(plans)

for i in plans:
    str = ''
    for j in i:
        str += '*'*j
        str +='|'
    print(str[:-1])



## 测试代码速度
import time
n= 20
k =5

matrix = create_matrix(n,k)
Tmatrix = init_Tmatrix(n,k)
plan_num, plans = split(n,k,matrix,Tmatrix)

beg_native = time.time()
print('原始法:',brute_split(n,k))
end_native = time.time()

print('原始法用时：',end_native-beg_native)

beg_dp = time.time()
print(f'动态规划法：有{n}个礼物，要分给{k} 个小朋友，分法总共有{plan_num}种')
end_dp = time.time()

print('动态规划法用时:',end_dp-beg_dp)
