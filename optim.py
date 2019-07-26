import torch as tc

def fitReg( reg, t, ft, maxiter=100, tol=1e-7 ):
    optimizer = tc.optim.LBFGS(reg.parameters(),lr=.01)
    criterion = tc.nn.MSELoss()

    def closure():
        optimizer.zero_grad()
        output = reg.forward(t)
        loss = criterion( output, ft )
        loss.backward()
        return loss
    
    for n in range(maxiter):
        optimizer.zero_grad()
        output = reg.forward(t)
        loss = criterion( output, ft )
        loss.backward()
        optimizer.step(closure)
        print( 'n = {}, loss = {}'.format(n,loss.item()) )

        if tc.abs(tc.abs(loss)).item() < tol:
            break

    return reg
