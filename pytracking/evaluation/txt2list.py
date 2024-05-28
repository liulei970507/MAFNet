txt_path = '/home/liulei/zhutianhao/gt.txt'
txt_path_1 = '/home/liulei/zhutianhao/all.txt'
with open(txt_path,'r') as f:
    lst = f.readlines()
for i in range(len(lst)):
    lst[i] = lst[i].replace('\n','').replace('.txt','')
# print(lst)

with open(txt_path_1,'r') as f:
    lst1 = f.readlines()
for i in range(len(lst1)):
    lst1[i] = lst1[i].replace('\n','').replace('.txt','')
a = list(set(lst1).difference(set(lst)))
print(a, len(a))