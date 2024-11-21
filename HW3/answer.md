### **Input**
The input is a \( 64 \times 64 \) image with 1 channel. This step does not introduce any trainable parameters.

### 2. **Conv1: 32 filters of size \( 5 \times 5 \), stride 1, padding 2**
Each filter has dimensions \( 5 \times 5 \) and processes 1 input channel. 

\[
\text{Parameters per filter} = 5 \times 5 \times 1 = 25
\]

Adding a bias term for each filter:
\[
\text{Total parameters for Conv1} = (25 + 1) \times 32 = 832
\]

---

### 3. **MaxPool1: \( 2 \times 2 \), stride 2**
Max pooling does not introduce trainable parameters.

---

### 4. **Conv2: 64 filters of size \( 3 \times 3 \), stride 1, padding 1**
Each filter has dimensions \( 3 \times 3 \) and processes 32 input channels (output channels of Conv1).

\[
\text{Parameters per filter} = 3 \times 3 \times 32 = 288
\]

Adding a bias term for each filter:
\[
\text{Total parameters for Conv2} = (288 + 1) \times 64 = 18,496
\]

---

### 5. **MaxPool2: \( 2 \times 2 \), stride 2**
Max pooling does not introduce trainable parameters.

---

### 6. **Flatten Layer**
After the second pooling layer, the feature map dimensions are reduced:
1. Input image size: \( 64 \times 64 \) 
2. After Conv1 and MaxPool1: \( \frac{64}{2} = 32 \) (output is \( 32 \times 32 \) with 32 channels)
3. After Conv2 and MaxPool2: \( \frac{32}{2} = 16 \) (output is \( 16 \times 16 \) with 64 channels)

Thus, the input to the fully connected layers is a flattened vector of size:
\[
16 \times 16 \times 64 = 16,384
\]

---

### 7. **Fully Connected 1: 1024 neurons**
Each neuron in this layer connects to the \( 16,384 \) input features.

\[
\text{Weights for FC1} = 16,384 \times 1024 = 16,777,216
\]

Adding biases for each neuron:
\[
\text{Total parameters for FC1} = 16,777,216 + 1024 = 16,778,240
\]

---

### 8. **Fully Connected 2: 10 neurons**
Each neuron in this layer connects to the 1024 outputs from the previous layer.

\[
\text{Weights for FC2} = 1024 \times 10 = 10,240
\]

Adding biases for each neuron:
\[
\text{Total parameters for FC2} = 10,240 + 10 = 10,250
\]

---

### **Total Trainable Parameters**
Adding up the trainable parameters from all layers:
\[
832 \, (\text{Conv1}) + 18,496 \, (\text{Conv2}) + 16,778,240 \, (\text{FC1}) + 10,250 \, (\text{FC2}) = 16,807,818
\]

Thus, the total number of trainable parameters in the network is:
\[
\boxed{16,807,818}
\]