import logging
import matplotlib.pyplot as plt


# 配置日志
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 设置 matplotlib 字体，解决中文显示问题
def setup_matplotlib():
    plt.rcParams['font.family'] = 'simsun'
    plt.rcParams['axes.unicode_minus'] = False

