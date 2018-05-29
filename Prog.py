def take_index(a,b,c,d):
    areas = [(a[i] * b[i], i) for i in range(len(a))]
    return sorted(areas)[-1][1]


A = [1,99,0,3,6,8]
B = [5,6,7,8,9,15]
C = [1,1,1,1,1,1]
D = [2,2,2,2,2,2]


print(take_index(A,B,C,D))