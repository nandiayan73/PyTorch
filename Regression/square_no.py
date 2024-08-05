import torch 

x=torch.tensor([[2,3,4,5]],dtype=torch.float32)
y=torch.tensor([[6,9,12,15]],dtype=torch.float32)

w=torch.tensor(2.0,dtype=torch.float32,requires_grad=True)

learning_rate=0.01
n_iters=50

# Defining the methods:

def forward(x):
    return w*x

def loss(y,y_pred):
    return ((y_pred-y)**2).mean()

# defining the network:

def trainNetwork(x,w):
    for epoch in range(0,n_iters):
        y_pred=forward(x)

        l=loss(y,y_pred)#calculates the loss

        l.backward() #calculates the gradient 

        # updating the weights

        with torch.no_grad():
            w-=learning_rate * w.grad

        w.grad.zero_()

        if(epoch%2==0):
            print(f"epoch= {epoch} ,loss={l:.3} ,w={w:.3}")
        if(epoch==n_iters-1):return w


def predict(test_set):
    wt=trainNetwork(x,w)
    res=wt*test_set
    return res

print(predict(torch.tensor(([4,5,6,8]),dtype=torch.float32)))




