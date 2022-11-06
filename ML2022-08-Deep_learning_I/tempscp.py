accuraciess_keep = []

for al in [0.001,0.05,0.1]:


    def ReLu(x):
        output = np.maximum(0,x)
        return output

    def Tanh(x):
        x_p = np.exp(x)
        x_m = np.exp(-x)
        output = (x_p - x_m) / (x_p + x_m)
        return output

    def Sigmoid(x):
        output = 1/(1+np.exp(-x))
        return output

    def Softmax(x):
        exp_x = np.exp(x)
        output = exp_x/np.sum(exp_x)
        return output

    h2 = 5
    h1 = 6
    W = [[], np.random.normal(0,0.1,[x_area,h1]),
             np.random.normal(0,0.1,[h1,h2]),
             np.random.normal(0,0.1,[h2,10])]
    B = [[], np.random.normal(0,0.1,[h1,1]),
             np.random.normal(0,0.1,[h2,1]),
             np.random.normal(0,0.1,[10,1])]

    act_funcs = [None, ReLu, Sigmoid, Softmax]

    L = len(W)-1

    def forward_layer(w, b, X, act_func):
        z = w.T * X + b
        if act_func is not None:
            y_hat = act_func(z)
        else:
            y_hat = z
        return z, y_hat

    def forward_one_step(X, W, B, act_funcs):
        L = len(W)-1
        a = [X]
        z = [[]]
        delta = [[]]
        dW = [[]]
        db = [[]]
        for l in range(1,L+1):
            z_layer, a_layer = None, None
            ### BEGIN SOLUTION
            z_layer, a_layer = forward_layer(W[l], B[l], a[l-1], act_funcs[l])
            ### END SOLUTION
            z.append(z_layer)
            a.append(a_layer)
            # Just to give arrays the right shape for the backprop step
            delta.append([]); dW.append([]); db.append([])
        return a, z, delta, dW, db
    
    def predict_y(W, b, X):
        M = X.shape[0]
        y_pred = np.zeros(M)
        for i in range(X.shape[0]):
            y_pred[i] = np.argmax(feed_forward(X[i,:].T, W, B, act_funcs))
        return y_pred

    def feed_forward(X,W,B, act_funcs):
        L = len(W)-1
        a = [X]
        z = [[]]
        delta = [[]]
        dW = [[]]
        db = [[]]
        for l in range(1,L+1):
            z_layer, a_layer = forward_layer(W[l], B[l], a[l-1], act_funcs[l])
            z.append(z_layer)
            a.append(a_layer)
        return a_layer

    def loss(y, yhat):
        l = - np.dot(y, np.log(yhat))
        return l

    def Linear_derivative(x):
        output = np.ones(x.shape)
        return output

    def ReLu_derivative(x):
        output = (x > 0) * 1
        return output

    def Tanh_derivative(x):
        t=Tanh(x)
        output = 1-t**2
        return output

    def Sigmoid_derivative(x):
        s=Sigmoid(x)
        output = np.multiply(s,1-s)
        return output

    def back_propagation(y, a, z, W, dW, db, act_deri):
        '''
        Backprop step. Note that derivative of multinomial cross entropy
        loss is the same as that of binary cross entropy loss. See
        https://levelup.gitconnected.com/killer-combo-softmax-and-cross-entropy-5907442f60ba
        for a nice derivation.
        '''
        L = len(W)-1

        delta[L] = a[L] - np.matrix(y_this).T
        for l in range(L,0,-1):
            db[l] = delta[l].copy()
            dW[l] = a[l-1] * delta[l].T
            if l > 1:
                delta[l-1] = np.multiply(act_deri[l-1](z[l-1]), W[l] * delta[l])

        return dW, db

    act_deri = [None, ReLu_derivative, Sigmoid_derivative, Softmax]

    def update_step(W, B, dW, db, alpha):
        L = len(W)-1
        for l in range(1,L+1):
            W[l] = W[l] - alpha * dW[l]
            B[l] = B[l] - alpha * db[l]
        return W, B


    cost_arr = [] 

    alpha = al
    max_iter = 100
    for iter in range(0, max_iter):
        loss_this_iter = 0
        # random index of m_train
        order = np.random.permutation(m_train)
        for i in range(0, m_train):
            # Grab the pattern order[i]
            x_this = X_train[order[i],:].T
            y_this = y_train[order[i],:]

            # Feed forward step
            a, z, delta, dW, db = forward_one_step(x_this, W, B, act_funcs)

            # calulate loss
            loss_this_pattern = loss(y_this, a[L])
            loss_this_iter = loss_this_iter + loss_this_pattern

            # back propagation
            dW, db = back_propagation(y_this, a, z, W, dW, db, act_deri)

            # update weight, bias
            W, B = update_step(W, B, dW, db, alpha)



        print('Epoch %d train loss %f' % (iter + 1, loss_this_iter[0,0]))

        cost_arr.append(loss_this_iter[0,0])
        
        if iter > 20:
            if cost_arr[-2] - cost_arr[-1] < 0.5:


                y_test_predicted = predict_y(W, B, X_test)
                y_correct = y_test_predicted == y_test_indices
                test_accuracy = np.sum(y_correct) / len(y_correct)
                accuraciess_keep.append(test_accuracy)

                print('Test accuracy: %.4f' % (test_accuracy))
                break
            elif iter > 95:
                y_test_predicted = predict_y(W, B, X_test)
                y_correct = y_test_predicted == y_test_indices
                test_accuracy = np.sum(y_correct) / len(y_correct)
                accuraciess_keep.append(test_accuracy)

                print('Test accuracy: %.4f' % (test_accuracy))
                break
                
    plt.plot(np.arange(len(cost_arr)), cost_arr,label = alpha)
    plt.title(f'Compare Learning rate')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    
plt.legend()
plt.show()
al = [0.001,0.05,0.1]
for al_each,accuraciess_keep_each in zip(al,accuraciess_keep):
    print(f'LR = {al_each} acc = {accuraciess_keep_each}')
    
    
    
    
 



#### NO OF NODE


accuraciess_keep = []
def ReLu(x):
    output = np.maximum(0,x)
    return output

def Tanh(x):
    x_p = np.exp(x)
    x_m = np.exp(-x)
    output = (x_p - x_m) / (x_p + x_m)
    return output

def Sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

def Softmax(x):
    exp_x = np.exp(x)
    output = exp_x/np.sum(exp_x)
    return output
for h1,h2 in zip([2,4,8],[2,3,6]):
    # h2 = 5
    # h1 = 6
    W = [[], np.random.normal(0,0.1,[x_area,h1]),
             np.random.normal(0,0.1,[h1,h2]),
             np.random.normal(0,0.1,[h2,10])]
    B = [[], np.random.normal(0,0.1,[h1,1]),
             np.random.normal(0,0.1,[h2,1]),
             np.random.normal(0,0.1,[10,1])]



    L = len(W)-1

    def forward_layer(w, b, X, act_func):
        z = w.T * X + b
        if act_func is not None:
            y_hat = act_func(z)
        else:
            y_hat = z
        return z, y_hat

    def forward_one_step(X, W, B, act_funcs):
        L = len(W)-1
        a = [X]
        z = [[]]
        delta = [[]]
        dW = [[]]
        db = [[]]
        for l in range(1,L+1):
            z_layer, a_layer = None, None
            ### BEGIN SOLUTION
            z_layer, a_layer = forward_layer(W[l], B[l], a[l-1], act_funcs[l])
            ### END SOLUTION
            z.append(z_layer)
            a.append(a_layer)
            # Just to give arrays the right shape for the backprop step
            delta.append([]); dW.append([]); db.append([])
        return a, z, delta, dW, db

    def feed_forward(X,W,B, act_funcs):
        L = len(W)-1
        a = [X]
        z = [[]]
        delta = [[]]
        dW = [[]]
        db = [[]]
        for l in range(1,L+1):
            z_layer, a_layer = forward_layer(W[l], B[l], a[l-1], act_funcs[l])
            z.append(z_layer)
            a.append(a_layer)
        return a_layer

    def loss(y, yhat):
        l = - np.dot(y, np.log(yhat))
        return l

    def Linear_derivative(x):
        output = np.ones(x.shape)
        return output

    def ReLu_derivative(x):
        output = (x > 0) * 1
        return output

    def Tanh_derivative(x):
        t=Tanh(x)
        output = 1-t**2
        return output

    def Sigmoid_derivative(x):
        s=Sigmoid(x)
        output = np.multiply(s,1-s)
        return output

    def back_propagation(y, a, z, W, dW, db, act_deri):
        '''
        Backprop step. Note that derivative of multinomial cross entropy
        loss is the same as that of binary cross entropy loss. See
        https://levelup.gitconnected.com/killer-combo-softmax-and-cross-entropy-5907442f60ba
        for a nice derivation.
        '''
        L = len(W)-1

        delta[L] = a[L] - np.matrix(y_this).T
        for l in range(L,0,-1):
            db[l] = delta[l].copy()
            dW[l] = a[l-1] * delta[l].T
            if l > 1:
                delta[l-1] = np.multiply(act_deri[l-1](z[l-1]), W[l] * delta[l])

        return dW, db

    act_deri = [None, ReLu_derivative, Sigmoid_derivative, Softmax]

    def update_step(W, B, dW, db, alpha):
        L = len(W)-1
        for l in range(1,L+1):
            W[l] = W[l] - alpha * dW[l]
            B[l] = B[l] - alpha * db[l]
        return W, B


    cost_arr = [] 

    alpha = 0.01
    max_iter = 100
    for iter in range(0, max_iter):
        loss_this_iter = 0
        # random index of m_train
        order = np.random.permutation(m_train)
        for i in range(0, m_train):
            # Grab the pattern order[i]
            x_this = X_train[order[i],:].T
            y_this = y_train[order[i],:]

            # Feed forward step
            a, z, delta, dW, db = forward_one_step(x_this, W, B, act_funcs)

            # calulate loss
            loss_this_pattern = loss(y_this, a[L])
            loss_this_iter = loss_this_iter + loss_this_pattern

            # back propagation
            dW, db = back_propagation(y_this, a, z, W, dW, db, act_deri)

            # update weight, bias
            W, B = update_step(W, B, dW, db, alpha)

        print('Epoch %d train loss %f' % (iter + 1, loss_this_iter[0,0]))
        cost_arr.append(loss_this_iter[0,0])
        if iter > 50:
            if cost_arr[-2] - cost_arr[-1] < 0.5:
                y_test_predicted = predict_y(W, B, X_test)
                y_correct = y_test_predicted == y_test_indices
                test_accuracy = np.sum(y_correct) / len(y_correct)
                accuraciess_keep.append(test_accuracy)
                print('Test accuracy: %.4f' % (test_accuracy))
                break
    plt.plot(np.arange(len(cost_arr)), cost_arr,label = f'nodes in h1 = {h1},nodes in h2={h2} ')
    plt.title('compare of no of node in layer')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
plt.legend()
plt.show()
h1 = [2,4,8]
h2 = [2,3,6]
for h1_each,h2_each,accuraciess_keep_each in zip(h1,h2,accuraciess_keep):
    print(f'nodes in h1 = {h1_each},nodes in h2={h2_each} acc = {accuraciess_keep_each}')