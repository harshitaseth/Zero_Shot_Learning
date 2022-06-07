import pickle as pkl
import matplotlib.pyplot as plt


train_loss = pkl.load(open("Final_train_loss.pkl","rb"))
val_loss = pkl.load(open("Final_val_loss.pkl","rb"))

plt.plot(train_loss)
plt.savefig("train_loss.png")

plt.plot(val_loss)
plt.savefig("val_loss.png")

