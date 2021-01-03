import random

graph = None
class node(object):
    def __init__(self,number=None):
        self.pre = None
        self.no = number
        self.label = []
        self.child = []
        self.cost = None

    def add_child(self,number):
        tmp_node = node(number=number)
        tmp_node.pre = self
        tmp_node.label=[i for i in self.label]
        tmp_node.label.append(number)
        tmp_node.cost= get_bound(tmp_node.label)
        self.child.append(tmp_node)

def get_bound(label):
    f = 0
    for i in range(0,len(label)-1):
        f = f+graph[label[i]-1][label[i+1]-1]
    if len(label)==len(graph):
        f = f+graph[label[-1]-1][0]
    return f

def f(N,d,x):
    return (x**(d+1) - (N+1)*x + N)**2
def df(N,d,x):
    return 2*f(N,d,x)*((d+1)*x**d-(N+1))
def ddf(N,d,x):
    return 2*df(N,d,x)*((d+1)*x**d-(N+1))+2*f(N,d,x)*((d+1)*d*x**(d-1))
def solve(N,d):
    x = 1.9
    delta = 1.0
    count = 0
    while abs(delta)>0.000001 and count<10000:
        delta = df(N,d,x)/(ddf(N,d,x)+0.00001)
        x = x - delta
        count = count + 1
    return x


def algorithm():
    tree = node(number=1)
    tree.label.append(1)
    tree.cost=0


    NN = len(graph)
    for idx in range(NN):
        graph[idx][idx] = float('inf')

    city = range(len(graph))
    city = set(city)
    # tree = node(number=1,n_city=len(city))
    tree = node(number=0)
    tree.label.append(0)
    tree.cost=0
    visit=[]
    paths = []
    count = 0
    fcnt = 0
    visit.append(tree)

    
    while len(visit)>0:
        if len(visit)==0:
            break
        N = visit[0]
        del(visit[0])

        if len(N.label)==(len(graph)):
            paths.append(N)
            # print(paths)
            fcnt = fcnt+1
            paths=sorted(paths,key= lambda x:x.cost)
            count = count+1
        child_list = set(city).difference(set(N.label))
        if len(child_list)==0:
            continue 
        for c in child_list:
            N.add_child(number=c)
            tmp = N.child
            for i in tmp:
                if i not in visit:
                  visit.append(i)
    print("RESULT:",paths[0].label,paths[0].cost)
    print("d=%d ,N= %d ,  b*=%f"%(NN-2,count,solve(count,NN-2)))
    print("ROUTEs:%d"%(fcnt))
    return paths[0].label, paths[0].cost        
