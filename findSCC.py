
import scipy.io
def canFinish(numCourses, prerequisites):
    indegree = [0] * numCourses
    adj = [[] for x in range(numCourses)]
    
    for prereq in prerequisites:
        adj[prereq[1]].append(prereq[0])
        indegree[prereq[0]] += 1

    queue = []
    for i in range(numCourses):
        if indegree[i] == 0:
            queue.append(i)
    
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for neighbor in adj[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    
    return numCourses == visited

branch_ = scipy.io.loadmat('MATPOWER/branch2.mat')['B']
line_to_nodes = [list( map(int,i) ) for i in branch_[:, 0:2]-1]
if canFinish(numCourses=8870, prerequisites=line_to_nodes):
    print("No cycle")
else:
    print("Cycle exists")
