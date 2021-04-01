import matplotlib.pyplot as plt
import seaborn as sns

# loss = [1.275, 1.187, 1.015, 0.9561, 0.8452, 0.9008, 0.8429, 0.8476 ,0.734, 0.7293, 0.77, 0.7011, 0.685, 0.6544, 0.6202, 0.6719, 0.6459, 0.6522, 0.5837, 0.5854, 0.5437, 0.5905, 0.558, 0.529]
# epochs = list(range(1, 25))

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# plt.plot(loss, 'black')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Showing How Loss Changes with Epochs')
# plt.grid(color='gray')
# ax.set_facecolor('#E5EBF5')
# # ax.set_facecolor((1.0, 0.47, 0.42))
# fig.savefig('Loss_graph.jpg')

# plt.show()

train = [0.7625026828609418, 0.7521452397614409, 0.7879443251369007, 0.7950646090005228, 0.838810257842309, 0.8393380186107047, 0.7787660763243628, 0.8615587928016308, 0.8331811176904809, 0.8235470947755319, 0.8291550249369463]
test = [0.7407423628102628, 0.74691142916013, 0.7585738920763634, 0.7657215422271973, 0.8014944936340951, 0.802001020515777, 0.7528605474186063, 0.8415156847686882, 0.7940638953259502, 0.7767015274835007, 0.8040651892304544]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(train, 'black')
plt.plot(test, 'r')
plt.grid(color='gray')
ax.set_facecolor('#E5EBF5')
plt.xlabel('Epoch')
plt.ylabel('mAP Score')
plt.title('Showing mAP Score vs Epoch')
plt.legend(['Train Set', 'Test Set'])
fig.savefig('mAP Score.jpg')
plt.show()
