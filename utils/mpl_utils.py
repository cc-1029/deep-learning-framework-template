import matplotlib as mpl
import matplotlib.pyplot as plt


def plt_display_chinese():
    """To show Chinese regularly in Matplotlib

    """
    # 设置字体
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    # 该语句解决图像中的 “-” 负号的乱码问题
    plt.rcParams['axes.unicode_minus'] = False