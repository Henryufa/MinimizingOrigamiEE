from pickle import TRUE
import numpy as np
import numpy.linalg as la

#what is length u, length v?? constant throughout so it should be a global variable
lengthu = 1
lengthv = 1

#decide theta and beta here
theta = np.pi/6
beta = np.pi/6


#node objects contain the vector value for the u, v, and w pointing to subsequent nodes
#they also have pointers for traversing next and previous u and v vectors to other nodes

class node:
    def __init__(self,u,v,w) -> None:
        self.uNext = None
        self.vNext = None
        self.uPrev = None
        self.vPrev = None
        self.u = u 
        self.v = v 
        self.w = w 
        #information about derivatives can be stored at the nodes to make this a dynamic programming solution
        self.dc = 0

#our larger structure keeps track of our base layer and the top/final node in our structure
class structure:
    def __init__(self) -> None:
        self.First_layer = []
        self.top = None

    #populates first layer of structure with an array of nodes
    def populate(self,nodes,theta,beta):
        #This is ok
        u = lengthu * np.array([np.sin(theta / 2), np.cos(theta / 2), 0])
        v = lengthv * np.array([-np.sin(theta / 2), np.cos(theta / 2), 0])

        # Define local basis
        t1 = np.array([1., 0., 0.])

        # Rotate dent by beta around t1
        u = (np.cos(beta) * (u - np.dot(u, t1) * t1) +
             np.sin(beta) * np.cross(t1, u) + np.dot(u, t1) * t1)
        v = (np.cos(beta) * (v - np.dot(v, t1) * t1) +
             np.sin(beta) * np.cross(t1, v) + np.dot(v, t1) * t1)
        
        w = u+v

        #adds first node to base layer
        self.First_layer.append(node(u,v,w))

        #points first node to the start of the next layer
        self.First_layer[0].vNext = node(0,0,0)

        #points the next layer back to the first layer
        self.First_layer[0].vNext.vPrev = self.First_layer[0]
        #print("/\\", end = " ")

        #appends a new node to first layer, points it to the second layer and points the second layer back to the first
        #does this for all but final node which will not have a vNext
        for i in range(1,nodes-1):
            self.First_layer.append(node(u,v,w))
            self.First_layer[i].uNext = self.First_layer[i-1].vNext
            self.First_layer[i].uNext.uPrev = self.First_layer[i]
            self.First_layer[i].vNext = node(0,0,0)
            self.First_layer[i].vNext.vPrev = self.First_layer[i]
            #print("/\\", end = " ")

        #adds final node without vNext
        self.First_layer.append(node(u,v,w))
        self.First_layer[nodes-1].uNext = self.First_layer[nodes-2].vNext
        self.First_layer[nodes-1].uNext.uPrev = self.First_layer[nodes-1]   
        #print("/\\")

    #builds the rest of the structure stopping when a layer of one node is built
    def construct(self,theta,beta):
        u = lengthu * np.array([np.sin(theta / 2), np.cos(theta / 2), 0])
        v = lengthv * np.array([-np.sin(theta / 2), np.cos(theta / 2), 0])

        # Define local basis
        t1 = np.array([1., 0., 0.])

        # Rotate dent by beta around t1
        u = (np.cos(beta) * (u - np.dot(u, t1) * t1) +
             np.sin(beta) * np.cross(t1, u) + np.dot(u, t1) * t1)
        v = (np.cos(beta) * (v - np.dot(v, t1) * t1) +
             np.sin(beta) * np.cross(t1, v) + np.dot(v, t1) * t1)
        
        w = u+v

        #define the start of the second layer
        start = self.First_layer[0].vNext

        #while we can build a layer of more than one node, build another layer until the previous layer has only two nodes
        while start.uPrev.vNext != None:

            #build layer passes back the new start node
            start = self.buildLayer(start,u,v,w)
        
        #print("/\\")
        start.u = u
        start.v = v

        #define the top of the structure
        self.top = start


    #first and last nodes in each layer are handled uniquely because the first node does not have a uNext pointer
    #the last node does not have a vNext pointer.
    def buildLayer(self,start,u,v,w) -> node:

        #initialize first node, don't give it a uNext, point it to the next layer, point next layer back to current
        start.u = u
        start.v = v
        start.w = w
        start.vNext = node(0,0,0)
        start.vNext.vPrev = start
        start.vNext.uPrev = start.uPrev.vNext
        cur_node = start.uPrev.vNext
        #print("/\\", end = " ")

        #construct new nodes but stop before constructing the last one in this layer
        while cur_node.uPrev.vNext != None:
            cur_node.u = u
            cur_node.v = v
            cur_node.w = w            
            cur_node.uNext = cur_node.vPrev.uNext.vNext
            cur_node.vNext = node(0,0,0)
            cur_node.vNext.vPrev = cur_node
            cur_node.vNext.uPrev = cur_node.uPrev.vNext
            cur_node = cur_node.uPrev.vNext
            #print("/\\", end = " ")

        #construct final node in this layer without a vNext pointer    
        cur_node.u = u
        cur_node.v = v
        cur_node.w = w            
        cur_node.uNext = cur_node.vPrev.uNext.vNext
        #print("/\\")

        #return the first node in the next layer
        return start.vNext

    def costTotal(self):

        #start at the first node in the structure
        cur_node = self.First_layer[0]
        cost = 0
        dcost_dtheta = 0
        start = cur_node

        #Iterate through layers until we are on the top layer
        while(cur_node != self.top):

            #Move right until we reach the final node in this layer
            while(cur_node.vNext != None):
                dcost_dtheta += dc_dtheta(cur_node)
                cost += energyfunc(cur_node.u,cur_node.v)
                cur_node = cur_node.vNext.uPrev

            #add the cost of the final node in this layer and move to the next layer
            dcost_dtheta += dc_dtheta(cur_node)
            cost += energyfunc(cur_node.u,cur_node.v)
            cur_node = start.vNext
            start = start.vNext

        #add the cost of the top node
        dcost_dtheta += dc_dtheta(cur_node)
        cost += energyfunc(start.u,start.v)
        return cost,dcost_dtheta

    #start at the top and recurse down getting info you need and coming back up 
    #def dC_dtheta(self):
    #    cur_node = self.top
    #    dC = dc_dtheta(cur_node)
    #    return dC

#takes the vectors of a node and computes cos(gamma)^2 where gamma is the angle between them
def energyfunc(u,v):
    a = np.dot(u,v)/lengthv/lengthu
    return a**2 

#takes a node and returns the derivative of its cost with respect to theta as a scalar        
def dc_dtheta(cur_node):

    #easy to calculate directly
    dcu = dc_du(cur_node.u,cur_node.v)
    dcv = dc_dv(cur_node.u,cur_node.v)

    #cannot get directly unless the node is on the base layer, usually must recurse
    dut = du_dtheta(cur_node)
    dvt = dv_dtheta(cur_node)

    #store dc at the node
    cur_node.dc = np.dot(dcu, dut) + np.dot(dcv, dvt)
    return cur_node.dc


def du1_dv1_dB(beta,m,n,t1):
#insert t1, m, and n as constants I don't know that they can be presumed constant, because they depend on theta

    #u = (np.cos(beta) * (m - np.dot(m, t1) * t1) +
    #     np.sin(beta) * np.cross(t1, m) + np.dot(m, t1) * t1)
    #v = (np.cos(beta) * (n - np.dot(n, t1) * t1) +
    #     np.sin(beta) * np.cross(t1, n) + np.dot(n, t1) * t1)
    

    du = np.array[(-np.sin(beta) * (m - np.dot(m, t1) * t1) +
         np.cos(beta) * np.cross(t1, m))]
    dv = np.array[(-np.sin(beta) * (n - np.dot(n, t1) * t1) +
         np.cos(beta) * np.cross(t1, n))]

    return du,dv

#w derivatives
def dw_du():
    return np.identity(3)
def dw_dv():
    return np.identity(3)

#takes a node and returns vector derivative
def dw_dtheta(cur_node):

    #if this node is on the base layer, calculate derivative directly
    if cur_node.uPrev == None:
        return dw1dt()

    #else we will have to recurse to previous layers for more information    
    return dw_du() @ du_dtheta(cur_node.uPrev) + dw_dv() @ dv_dtheta(cur_node.vPrev)

#calculate dw/dtheta directly    np.identity(3) @ 
def dw1dt():
    return dw_dv() @ dv1dt() + dw_du() @ du1dt()


#v derivatives
def dv_du():
    return np.identity(3)
def dv_dw():
    return np.identity(3)


#takes a node and returns vector derivative 
def dv_dtheta(cur_node):
    
    #if this node is on the base layer, calculate dv/dtheta directly
    if cur_node.uPrev == None:
        return dv1dt()

    #else we will have to recurse to previous layers
    return dv_dw() @ dw_dtheta(cur_node.uPrev) + dv_du() @ du_dtheta(cur_node.uPrev)

def dv1dt():
    return np.array([-np.cos(theta/2)/2,-np.sin(theta/2)/2,0])

#u derivatives
def du_dw():
    return np.identity(3)
def du_dv():
    return np.identity(3)

#takes a node and returns vector derivative 
def du_dtheta(cur_node):

    #if the node is in the base layer du/dtheta can be calculated directly
    if cur_node.vPrev == None:
        return du1dt()

    #else we must recurse to previous layers to get more information
    return du_dw() @ dw_dtheta(cur_node.vPrev) + du_dv() @ dv_dtheta(cur_node.vPrev)

def du1dt():
    return np.array([np.cos(theta/2)/2,-np.sin(theta/2)/2,0])


def dc_du(u,v):
    return v        #2 * np.dot(u,v)/lengthv/lengthu * (v/lengthv/lengthu - np.dot(np.dot(u,u),v)/lengthu**3/lengthv)   

def dc_dv(u,v):
    return u        #2 * np.dot(u,v)/lengthv/lengthu * (u/lengthv/lengthu - np.dot(np.dot(v,v),u)/lengthv**3/lengthu)


origami = structure()
nodes = 4

#structured centered with respect to original theta
origami.populate(nodes,theta,beta)
origami.construct(theta,beta)
cost1,cost_dtheta1 = origami.costTotal()

#important values for original structure
print("\ncost1:  " + str(cost1))
print("cost_dtheta1:  " + str(cost_dtheta1))
print("top.dc:  " + str(origami.top.dc))
print("top.uPrev.dc:  " + str(origami.top.uPrev.dc))


#back and forward structures for calculating centered finite difference
dt = -.005
theta_b = theta + dt
origami_b = structure()
origami_b.populate(nodes,theta_b,beta)
origami_b.construct(theta_b,beta)
cost_b,cost_dtheta_b = origami_b.costTotal()

dt = -dt
theta_f = theta + dt
origami_f = structure()
origami_f.populate(nodes,theta_f,beta)
origami_f.construct(theta_f,beta)
cost_f,cost_dtheta_f = origami_f.costTotal()


print("finite difference:  " + str((cost_f-cost_b)/2/dt) + "\n")
