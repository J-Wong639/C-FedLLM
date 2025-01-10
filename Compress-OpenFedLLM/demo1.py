import numpy as np
import torch 



delta_w=torch.randn(20,10)

# print(type(delta_w))
# print(delta_w)

np_delta_w=delta_w.detach().numpy()

# print(type(np_delta_w))
# print("np_delta_w=",np_delta_w)
# print(np_delta_w.shape)
# print(np_delta_w)


U,S,VT=np.linalg.svd(np_delta_w)


print("shape U=",U.shape)
print("shape VT=",VT.shape)

r=3

P=U[:,:r]
print("shape P=",P.shape) # (20,3) 对应 (10,20)
Q=VT.T[:,:r]

print("shape Q=",Q.shape) # (10,3) 对应 (20,10)

# print("P=",P)


# np_delta_w=np.dot(P.T,np_delta_w)

np_delta_w=np.dot(np_delta_w,Q)

# print("np_delta_w=",np_delta_w)
print(np_delta_w.shape)