# import numpy as np
import torch
# a=np.array([[1,2,3,4]],dtype=float)
# b=np.array([[2,4,6,8]],dtype=float)


# #calculating the mean manually
# c=((b-a)**2)
# sum=0
# for i in range(0,4):
#     sum+=c[0][i]
# print(sum/4)

# # Calculatign the mean with numpy:
# print(c.mean())

def loss(y_pred,y):
    return ((y_pred-y)**2).mean()
w= torch.tensor(0.0,dtype=torch.float32,requires_grad=True)
y= torch.tensor([1,2,3,4],dtype=torch.float32)
# y_pred= torch.tensor([2,2.4,5,6],dtype=torch.float32)

y_pred=w*y

l=loss(y_pred,y)
# print(l)
l.backward()
print(w.grad)


