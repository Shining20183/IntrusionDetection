import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

class Visual(object):
    def cm_plot(self,original_label, predict_label,labels, pic=None):
        cm = confusion_matrix(original_label, predict_label)  # 由原标签和预测标签生成混淆矩阵

        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels,rotation=45)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        # plt.title('Confusion Matrix')
        plt.colorbar()

        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()  #使保存的图片完整
        if pic is not None:
            plt.savefig(str(pic) + '.jpg')
            plt.show()

        plt.show()