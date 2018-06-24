
# coding: utf-8

# In[22]:


#Manual Neural Net


# In[23]:


import numpy as np


# In[24]:


class Operation():
    
    def __init__(self, input_nodes=[]):
        
        self.input_nodes = input_nodes
        self.output_nodes = []
    
        for node in input_nodes:
            node.output_nodes.append(self)
            
        _default_graph.operations.append(self)
        
    def compute(self):
        pass


# In[25]:


class add(Operation):
    
    def __init__(self,x,y):
        
        super().__init__([x,y])
    
    def compute(self,x_var,y_var):
        self.inputs = [x_var,y_var]
        return x_var + y_var


# In[26]:


class multiply(Operation):
    
    def __init__(self,x,y):
        
        super().__init__([x,y])
    
    def compute(self,x_var,y_var):
        self.inputs = [x_var,y_var]
        return x_var * y_var    


# In[27]:


class matmul(Operation): #Matrix multiplication
    
    def __init__(self,x,y):
        
        super().__init__([x,y])
    
    def compute(self,x_var,y_var):
        
        self.inputs = [x_var,y_var]
        return x_var.dot(y_var)


# In[28]:


class Placeholder(): #The x variable
    
    def __init__(self):
        
        self.output_nodes = []
        
        _default_graph.placeholders.append(self)
    


# In[29]:


class Variable(): #Coefficiants or constants
    
    def __init__(self, initial_value=None):
    
        self.value = initial_value
        self.output_nodes = []

        _default_graph.variables.append(self)


# In[30]:


class Graph(): #Acts as a "storage room" for variables, operations, placeholders
               # That's why _default_graph is a global variable
    
    def __init__(self):
        
        self.operations = []
        self.placeholders = []
        self.variables = []
    
    def set_as_default(self):
        
        global _default_graph
        _default_graph = self


# In[31]:


# z = Ax + b

# A = 10

# b = 1

# z = 10x + 1


# In[32]:


g = Graph()


# In[33]:


g.set_as_default()


# In[34]:


A = Variable(10)
b = Variable(1)


# In[35]:


x = Placeholder()


# In[36]:


y = multiply(A,x)


# In[37]:


z = add(y,b)


# In[44]:


def traverse_postorder(operation):
    
    nodes_postorder = []
    
    def recurse(node): #post order tree traversal
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node) #Gets nodes in "correct" order
    
    recurse(operation)
    return nodes_postorder
    


# In[45]:


class Session(): #takes in variables, placeholders and solves problem
    
    def run(self, operation, feed_dict={}):
        
        nodes_postorder = traverse_postorder(operation)
        
        for node in nodes_postorder:
            
            if type(node) == Placeholder:
                
                node.output = feed_dict[node]
            
            elif type(node) == Variable:
                
                node.output = node.value
                
            else:
                
                node.inputs = [input_node.output for input_node in node.input_nodes]
                
                node.output = node.compute(*node.inputs)
            
            if type(node.output) == list:
                node.output = np.array(node.output)
                
        return operation.output
                


# In[46]:


sess = Session()


# In[47]:


result = sess.run(operation=z, feed_dict={x:10})


# In[48]:


print(result)


# In[ ]:




