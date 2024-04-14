import cupy as cp

# this correctly shows how to concatenate cp.arrays()

cp_current_nodes = cp.empty((0, 3), dtype=cp.float32)  # stores [[shrink_gateway, lat, long],...]
row = cp.array([ 1.6158424e+09,  2.1919000e+04,  2.9741098e+01, -9.5533028e+01])
row2 = cp.array([ 9.000424e+09,  8.777000e+04,  7.900098e+01, -6.2223028e+01])
row3 = cp.array([ 2.000424e+09,  4.777000e+04,  -1.900098e+01, -96.2223028e+01])

#cp_current_nodes = cp.array(row[1:])

print(f"row[None,:]: {row[None,:]}")
foo = cp.concatenate((row[None,:],row2[None,:]),axis=0)
print(f"concat of row and row2: {foo}") # That's it!!
print(f"row after concat of row and row2 {row}")

foo2 = cp.append(row[None,:],row2[None,:],axis=0)
print(f"row after append of row2 to row {foo2}")

print(f"cp_current_nodes before: {cp_current_nodes} \n {row}")
new_arr = cp.concatenate((cp_current_nodes[None,:], cp.array([row[1:],])),axis=0)  # TODO: maybe this one isn't working??
print(f"cp_current_nodes intermediate: {new_arr} \n {row2}")

cp.concatenate((cp_current_nodes, row2[1:]))

print(f"cp_current_nodes after: {cp_current_nodes}")

