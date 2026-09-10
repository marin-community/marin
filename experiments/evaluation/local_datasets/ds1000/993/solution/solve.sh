#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
'''
training part
'''
# X, Y = load_iris(return_X_y=True)
# lossFunc = torch.nn.CrossEntropyLoss()
# opt = torch.optim.Adam(MyNet.parameters(), lr=0.001)
# for batch in range(0, 50):
#     for i in range(len(X)):
#         x = MyNet(torch.from_numpy(X[i]).float()).reshape(1, 3)
#         y = torch.tensor(Y[i]).long().unsqueeze(0)
#         loss = lossFunc(x, y)
#         loss.backward()
#         opt.step()
#         opt.zero_grad()
#         # print(x.grad)
#         # print(loss)
#     # print(loss)
output = MyNet(input)
probs = torch.nn.functional.softmax(output.reshape(1, 3), dim=1)
confidence_score, classes = torch.max(probs, 1)
PY
