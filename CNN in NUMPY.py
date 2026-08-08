import numpy as np
from keras.datasets import mnist

def convolve2d(image, filter,stride = 1,padding = 0):
    
    image = np.pad(image, ((0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)
    
    k_h, k_w = filter.shape 
    batch_size = image.shape[0]

    output_h = (image.shape[1] - k_h) // stride + 1
    output_w = (image.shape[2] - k_w) // stride + 1
    
    output = np.zeros((batch_size, output_h, output_w)) 

    for i in range (output_h):
        for j in range (output_w):
            
            patch = image[:, i * stride :i * stride + k_h,
            j * stride :j * stride + k_w ]
            
            output[:,i, j]  = np.sum(patch * filter, axis =(1,2))
     
    return output
    


    
def convolve2d_multi(image, filters, stride = 1,padding = 0):

    feature_maps = []
    for f in filters :
        feature_map = convolve2d(image, f, stride, padding)
        feature_maps.append(feature_map)
    return np.stack(feature_maps, axis = 1)



def relu(feature_maps):

    return np.maximum(0, feature_maps)



def poolmax2d(feature_maps, pool_size = 2, stride = 2):

    batch_size, num_maps, h, w = feature_maps.shape

    output_h = (h - pool_size) // stride + 1
    output_w = (w - pool_size) // stride + 1

    x = np.zeros((batch_size, num_maps, output_h, output_w))

    for m in range(num_maps):
        for i in range(output_h):
            for j in range(output_w):
                patch = feature_maps[:, m, i * stride : i * stride + pool_size,
                                      j * stride : j * stride + pool_size]

                x[:,m, i , j] = np.max(patch, axis = (1,2))
    return x



def softmax(x):
    x = x - np.max(x, axis = 1, keepdims = True)
    return np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)



def forward(image, filters, W1,W2,b1,b2):

    conv_out = convolve2d_multi(image, filters, stride=1, padding=0)
    relu_out = relu(conv_out)
    pool_out = poolmax2d(relu_out, pool_size=2, stride=2)

    x = pool_out.reshape(pool_out.shape[0], -1)

    layer1 = x @ W1 + b1
    layer1_relu = np.maximum(0, layer1)

    layer2 = layer1_relu @ W2 + b2

    output = softmax(layer2)


    return conv_out, relu_out, pool_out, x, layer1, layer1_relu, layer2, output
    





def backward(image, x, filters, relu_out, W1, W2, b1, b2,
              layer1_relu, output, target,pool_out):   

    batch_size = output.shape[0]

    dlayer2 = output - target

    dW2 = (layer1_relu.T @ dlayer2)
    db2 = np.sum(dlayer2, axis = 0)

    dlayer1_relu = dlayer2 @ W2.T 

    relu_mask = (layer1_relu > 0).astype(float)
    dlayer1 = dlayer1_relu  * relu_mask

    dW1 = x.T @ dlayer1
    db1 = np.sum(dlayer1, axis = 0)

    dx = dlayer1 @ W1.T

    dx = dx.reshape(pool_out.shape)


    drelu_out = np.zeros_like(relu_out)
    batch_size, num_maps, h, w = dx.shape

    for m in range(num_maps):
        for i in range(h):
            for j in range(w):

                patch = relu_out[:, m, i *2 : i * 2 + 2,
                                    j *2 : j * 2 + 2]

                max_vals = np.max(patch, axis=(1, 2))[:, np.newaxis, np.newaxis]
                mask = (patch == max_vals).astype(float)  

                drelu_out[:, m, i *2 : i * 2 + 2, j *2 : j * 2 + 2] +=  mask * dx[:, m, i, j,
                                                                                  np.newaxis, np.newaxis]


    dconv_out = drelu_out * (relu_out > 0).astype(float)
    dfilters = np.zeros_like(filters)

    num_filters, h, w = filters.shape
    _, _, H, W = dconv_out.shape 

    for m in range(num_filters):
        for i in range(H):
            for j in range(W):

                patch = image[:, i : i + h,
                               j : j + w]

                dfilters[m] += np.sum(patch * dconv_out[:, m, i, j, np.newaxis, np.newaxis],
                                               axis = 0 )

    return dW2, db2, dW1, db1, dfilters



def gradient_descent(W1,b1,W2,b2,filters,
                     dW1,db1,dW2,db2,dfilters,batch_size,
                     lr = 0.01):

    W1 -= lr * (dW1/batch_size)
    b1 -= lr * (db1/batch_size)

    W2 -= lr * (dW2/batch_size)
    b2 -= lr * (db2/batch_size)

    filters -= lr * (dfilters/batch_size)

    return W1 ,b1, W2, b2, filters





(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0


def one_hot(y,num_classes=10):

    result = np.zeros((len(y), num_classes))
    result[np.arange(len(y)), y] = 1

    return result 

Y_train = one_hot(Y_train)
Y_test = one_hot(Y_test)



np.random.seed(0)

filters = np.random.randn(8,3,3) * 0.1
W1 = np.random.randn(1352,128) * 0.01
b1 = np.zeros(128)
W2 = np.random.randn(128,10) * 0.01
b2 = np.zeros(10)

batch_size = 32
epochs = 10

for epoch in range(epochs):

   indices = np.random.permutation(len(X_train))
   X_train_shuffled = X_train[indices]
   Y_train_shuffled = Y_train[indices]

   total_loss = 0

   for i in range(0, len(X_train), batch_size):

     X_batch = X_train_shuffled[i : i + batch_size]
     Y_batch = Y_train_shuffled[i : i + batch_size]

     image = X_batch
     target = Y_batch


     conv_out, relu_out, pool_out, x, layer1, layer1_relu, layer2, output = forward(image, filters, W1, W2, b1, b2)

     loss = -np.mean(np.sum(Y_batch * np.log(output + 1e-8), axis=1))
     total_loss += loss

     dW2, db2, dW1, db1, dfilters = backward(image, x, filters, relu_out, W1, W2, b1, b2, layer1_relu, output, target,pool_out)

     W1, b1, W2, b2, filters = gradient_descent(W1, b1, W2, b2, filters, dW1, db1, dW2, db2, dfilters,batch_size, lr=0.01)
 


   print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(X_train):.4f}")



# TEST 
correct = 0

for i in range(0, len(X_test), batch_size):
    X_batch = X_test[i : i + batch_size]
    Y_batch = Y_test[i : i + batch_size]

    _, _, _, _, _, _, _, output = forward(X_batch, filters, W1, W2, b1, b2)

    predictions = np.argmax(output, axis=1)
    correct += np.sum(predictions == np.argmax(Y_batch, axis = 1))

accuracy = (correct / len(X_test)) * 100
print(f"Test Accuracy: {accuracy:.2f}%")