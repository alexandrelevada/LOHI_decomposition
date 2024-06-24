"""
    Network filtering and decomposition using information geometry in the Potts MRF model

    Python code for the first set of experiments of the paper: k-NN networks from datasets

    Author: Alexandre L. M. Levada
"""
import warnings
import matplotlib as mpl
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
import sklearn.neighbors as sknn
import sklearn.datasets as skdata
import sklearn.utils.graph as sksp
from scipy import stats
from sklearn import preprocessing
from sklearn import metrics
from numpy import inf
from scipy import optimize
from scipy.signal import medfilt
from networkx.convert_matrix import from_numpy_array
from sklearn.model_selection import train_test_split
from seaborn import kdeplot

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

# Optional function to normalize the curvatures to the interval [0, 1]
def normalize_curvatures(curv):
    if curv.max() != curv.min():
        k = 0.001 + (curv - curv.min())/(curv.max() - curv.min())
    else:
        k = curv
    return k

# Build the KNN graph
def build_KNN_Graph(dados, k):
    # Generate KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='distance')
    # Computes geodesic distances
    A = knnGraph.toarray()  
    return A

# Plot the KNN graph
def plot_KNN_graph(A, target, K=0, pos=0, layout='spring'):
	# Create a networkX graph object
	G = from_numpy_array(A)
	color_map = []
	for i in range(A.shape[0]):
	    if type(K) == list:
	        if K[i] > 0:
	    	    color_map.append('black')
	        else:
		        if target[i] == 0:
			        color_map.append('blue')
		        elif target[i] == 1:
			        color_map.append('red')
		        elif target[i] == 2:
			        color_map.append('green')
		        elif target[i] == 3:
			        color_map.append('purple')
		        elif target[i] == 4:
			        color_map.append('orange')
		        elif target[i] == 5:
			        color_map.append('magenta')
		        elif target[i] == 6:
			        color_map.append('darkkhaki')
		        elif target[i] == 7:
			        color_map.append('brown')
		        elif target[i] == 8:
			        color_map.append('salmon')
		        elif target[i] == 9:
			        color_map.append('cyan')
		        elif target[i] == 10:
			        color_map.append('darkcyan')    
	    else:
	        if target[i] == 0:
	            color_map.append('blue')
	        elif target[i] == 1:
	            color_map.append('red')
	        elif target[i] == 2:
	            color_map.append('green')
	        elif target[i] == 3:
	            color_map.append('purple')
	        elif target[i] == 4:
	        	color_map.append('orange')
	        elif target[i] == 5:
	            color_map.append('magenta')
	        elif target[i] == 6:
	            color_map.append('darkkhaki')
	        elif target[i] == 7:
	            color_map.append('brown')
	        elif target[i] == 8:
	            color_map.append('salmon')
	        elif target[i] == 9:
	            color_map.append('cyan')
	        elif target[i] == 10:
	        	color_map.append('darkcyan')
	plt.figure(1)
	# Several layouts to choose, here we prefer the spring and kamada-kawai layouts  
	if np.isscalar(pos):
	    if layout == 'spring':
	        pos = nx.spring_layout(G)
	    else:
	        pos = nx.kamada_kawai_layout(G) # ideal para plotar a árvore!
	nx.draw_networkx(G, pos, node_size=50, node_color=color_map, with_labels=False, width=0.25, alpha=0.4)
	plt.show()
	return pos

# Compute the free energy of a Potts MRF model outcome
def free_energy():
	n = A.shape[0]
	free_energy = 0
	for i in range(n):
		neighbors = A[i, :]
		indices = neighbors.nonzero()[0]
		labels = target[indices]
		uim = np.count_nonzero(labels==target[i])
		free_energy += uim
	return free_energy

# Defines the pseudo-likelihood function for a Potts MRF model outcome 
def pseudo_likelihood(beta):
	n = A.shape[0]
	# Computes the free energy
	free = free_energy()
	# Computes the number of labels (states of the Potts model)
	c = len(np.unique(target))
	# Computes the expected energy
	expected = 0
	for i in range(n):
		neighbors = A[i, :]
		indices = neighbors.nonzero()[0]
		labels = target[indices]
		num = 0
		den = 0
		for k in range(c):
			u = np.count_nonzero(labels==k)
			e = np.exp(beta*u)
			num += u*e
			den += e
		expected += num/den
	# Calculates the PL function value
	PL = free - expected
	return PL

# Compute the local first and second order Fisher local information (for each node of the network)
def FisherInformation(A, beta):
	n = A.shape[0]
	# Computes the number of labels (states of the Potts model)
	c = len(np.unique(target))
	PHIs = np.zeros(n)
	PSIs = np.zeros(n)
	for i in range(n):
		neighbors = A[i, :]
		indices = neighbors.nonzero()[0]
		labels = target[indices]
		uim = np.count_nonzero(labels==target[i])
		Uis = np.zeros(c)
		vi =  np.zeros(c)
		wi = np.zeros(c)
		Ai = np.zeros((c, c))
		Bi = np.zeros((c, c))
		# Build vectors vi and wi
		for k in range(c):
			Uis[k] = np.count_nonzero(labels==k)
			vi[k] = uim - Uis[k]
			wi[k] = np.exp(beta*Uis[k])
		# Build matrix A
		for k in range(c):
			Ai[:, k] = Uis
		# Build matrix B
		for k in range(c):
			for l in range(c):
				Bi[k, l] = Uis[k] - Uis[l]  
		# Compute the first and second order Fisher information
		PHIs[i] = np.sum( np.kron((vi*wi), (vi*wi).T) ) / np.sum( np.kron(wi, wi.T) )
		Li = Ai*Bi
		Mi = np.reshape(np.kron(wi, wi.T), (c, c))
		PSIs[i] = np.sum( Li*Mi ) / np.sum( np.kron(wi, wi.T) )
	return (PHIs, PSIs)

# Community detection
def community_detection(A, target):
    G = nx.from_numpy_array(A)
    # Clauset_Newman-Moore
    Comu = nx.community.greedy_modularity_communities(G)
    # Louvain 
    #Comu = nx.community.louvain_communities(G)
    r = 0
    labels = target.copy()
    for c in Comu:
        labels[list(c)] = r
        r += 1
    return labels

# Computes the evaluation metrics
def compute_metrics(A, labels):
    rotulos = np.unique(labels)    
    comunidades = []
    for c in rotulos:        
        community = np.where(labels==c)[0]
        comunidades.append(community)
    G = nx.from_numpy_array(A)
    modula = nx.community.modularity(G, comunidades)
    coverage, performance = nx.community.partition_quality(G, comunidades)
    cc = nx.average_clustering(G)
    return (modula, coverage, performance, cc)


##############################################
############# Beginning of the script
##############################################
#X = skdata.load_iris()
#X = skdata.load_wine()
#X = skdata.load_breast_cancer()
#X = skdata.load_digits()
#X = skdata.fetch_openml(name='prnn_crabs', version=1) 
#X = skdata.fetch_openml(name='Smartphone-Based_Recognition_of_Human_Activities', version=1)    

#X = skdata.fetch_openml(name='texture', version=1)     
#X = skdata.fetch_openml(name='segment', version=1)     
#X = skdata.fetch_openml(name='mfeat-karhunen', version=1)
#X = skdata.fetch_openml(name='mfeat-pixel', version=1)    
#X = skdata.fetch_openml(name='mfeat-fourier', version=1)
#X = skdata.fetch_openml(name='tecator', version=2)
#X = skdata.fetch_openml(name='satimage', version=1)
#X = skdata.fetch_openml(name='pendigits', version=1)   
X = skdata.fetch_openml(name='optdigits', version=1)  
#X = skdata.fetch_openml(name='MNIST_784', version=1)
#X = skdata.fetch_openml(name='Kuzushiji-MNIST', version=1)
#X = skdata.fetch_openml(name='USPS', version=1) 
#X = skdata.fetch_openml(name='JapaneseVowels', version=1)
#X = skdata.fetch_openml(name='vowel', version=1)

# Define the layout to plot the network
LAYOUT = 'spring'
#LAYOUT = 'kawai'

dados = X['data']
target = X['target']

# Convert labels to integers
S = list(set(target))
target = np.array(target)
for i in range(len(target)):
	for k in range(len(S)):
		if target[i] == S[k]:
			target[i] = k
		else:
			continue

# Convert to integers
target = target.astype('int32')
target_orig = target.copy()

# Reduce large datasets
if dados.shape[0] >= 50000:
    dados, _, target, _ = train_test_split(dados, target, train_size=0.025, random_state=42)
elif dados.shape[0] >= 10000:
    dados, _, target, _ = train_test_split(dados, target, train_size=0.1, random_state=42)
elif dados.shape[0] >= 5000:
    dados, _, target, _ = train_test_split(dados, target, train_size=0.2, random_state=42)
elif dados.shape[0] > 2200:
    dados, _, target, _ = train_test_split(dados, target, train_size=0.5, random_state=42)

# For opemML datasets - categorical data 
if not isinstance(dados, np.ndarray):
    cat_cols = dados.select_dtypes(['category']).columns
    dados[cat_cols] = dados[cat_cols].apply(lambda x: x.cat.codes)
    # Convert to numpy (openml uses dataframe)
    dados = dados.to_numpy()

n = dados.shape[0]
m = dados.shape[1]
c = len(np.unique(target))

print('N = ', n)
print('M = ', m)
print('C = %d' %c)

# Estimates the number of neighbors
#nn = round(np.sqrt(n))
# Fixed number of neighbors
nn = 15

print('K = ', nn)
print()

# Remove nan's
dados = np.nan_to_num(dados)
# Data standardization (to deal with variables having different units/scales)
dados = preprocessing.scale(dados)
# Build the adjacency matrix of the graph
A = build_KNN_Graph(dados, nn)

# Plot the original network
pos = plot_KNN_graph(A, target, layout=LAYOUT)
modula, coverage, performance, cc = compute_metrics(A, target)
print('Modularity (original network): ', modula)
print('Coverage (original network): ', coverage)
print('Performance (original network): ', performance)
print('Avg clustering coeficient (original network): ', cc)
print()

comunidades = []
for c in np.unique(target):
    comunidades.append(np.where(target==c)[0])

G = nx.from_numpy_array(A)
print('Number of nodes: ', len(G.nodes()))
print('Number of edges: ', len(G.edges()))
print()
condutancias = []
for i in range(len(comunidades)):
    for j in range(i+1, len(comunidades)):
        cond = nx.cuts.conductance(G, comunidades[i], comunidades[j])
        condutancias.append(cond)
print('Average conductancea between communities in the network: %f' %(sum(condutancias)/len(condutancias)))
print('Maximum conductancea between communities in the network: %f' %(max(condutancias)))
print()

# Estimates the maximum pseudo-likelihood estimator of the inverse temperature
#sol = optimize.root_scalar(pseudo_likelihood, bracket=[-5, 5], method='brentq')
sol = optimize.root_scalar(pseudo_likelihood, method='secant', x0=0, x1=1)
print('MPL estimator: ', sol)
print()
# Maximum pseudo-likelihood estimator
beta_mpl = sol.root

# Compute the first and second order local Fisher information 
PHI, PSI = FisherInformation(A, beta_mpl)
# Approximate the local curvatures
curvaturas = -PSI/(PHI+0.001)
# Normalize curvatures
K = normalize_curvatures(curvaturas)
# Threshold
limiar = np.quantile(K, 0.75)
for i in range(n):
    if K[i] < limiar:
        K[i] = 0

# Plot high information points
pos = plot_KNN_graph(A, target, K=list(K), pos=pos, layout=LAYOUT)

# Network decomposition
H_nodes = np.where(K>0)[0]
L_nodes = np.where(K==0)[0]
labels_H = target[H_nodes]
labels_L = target[L_nodes]

# Conductance
G = nx.from_numpy_array(A)
cond = nx.cuts.conductance(G, L_nodes, H_nodes)
print('Conductance between L and H nodes: ', cond)
print()

# Remove low information nodes
G = nx.from_numpy_array(A)
G.remove_nodes_from(H_nodes)
L_subgraph = nx.to_numpy_array(G)
modula, coverage, performance, cc = compute_metrics(L_subgraph, labels_L)
print('Modularity (L-subgraph): ', modula)
print('Coverage (L-subgraph): ', coverage)
print('Performance (L-subgraph): ', performance)
print('Avg clustering coeficient (L-subgraph): ', cc)
print()

# Plot the L-subgraph
pos = plot_KNN_graph(L_subgraph, labels_L, layout=LAYOUT)

# Plot spectrum
L_spectrum = np.real(nx.laplacian_spectrum(G))

# Remove high information nodes
G = nx.from_numpy_array(A)
G.remove_nodes_from(L_nodes)
H_subgraph = nx.to_numpy_array(G)
modula, coverage, performance, cc = compute_metrics(H_subgraph, labels_H)
print('Modularity (H-subgraph): ', modula)
print('Coverage (H-subgraph): ', coverage)
print('Performance (H-subgraph): ', performance)
print('Avg clustering coeficient (H-subgraph): ', cc)

# Plot the H-subgraph
pos = plot_KNN_graph(H_subgraph, labels_H, layout=LAYOUT)

# Plot spectrum
H_spectrum = np.real(nx.laplacian_spectrum(G))

# Plot the spectrums
f = plt.figure(1)
kdeplot(L_spectrum, bw_adjust=0.25, color='blue')
plt.title('L-subgraph spectrum')
f.show()

g = plt.figure(2)
kdeplot(H_spectrum, bw_adjust=0.25, color='red')
plt.title('H-subgraph spectrum')
g.show()
