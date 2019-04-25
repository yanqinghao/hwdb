import matplotlib.pyplot as plt
import pandas as pd

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

res = pd.read_csv('./result/res.csv')
res_acc = res[['acc','val_acc']]
res_loss = res[['loss','val_loss']]
smooth_acc_history = smooth_curve(res_acc.values)
plt.plot(range(1, len(smooth_acc_history) + 1), smooth_acc_history)
plt.xlabel('Epochs')
plt.ylabel('Training and validation accuracy')
# plt.show()
plt.savefig('./result/smooth_acc_100.png')
plt.figure()
smooth_loss_history = smooth_curve(res_loss.values)
plt.plot(range(1, len(smooth_loss_history) + 1), smooth_loss_history)
plt.xlabel('Epochs')
plt.ylabel('Training and validation loss')
# plt.show()
plt.savefig('./result/smooth_loss_100.png')
