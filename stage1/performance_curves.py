import numpy as np
import matplotlib.pyplot as plt

train_loss = np.loadtxt("train_loss",delimiter=',')
val_loss = np.loadtxt("val_loss",delimiter=',')
epoch_num = np.arange(1,len(train_loss)+1)

plt.figure(0)
plt.plot(epoch_num,train_loss,label="Training Loss")
plt.plot(epoch_num,val_loss,label="Validation Loss")
pltl.title("")
plt.xlabel("Epoch Number")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()
