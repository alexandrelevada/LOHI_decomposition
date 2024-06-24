"""
    Network filtering and decomposition using information geometry in the Potts MRF model

    Python code for the second set of experiments of the paper: network data

    Author: Alexandre L. M. Levada
"""
import warnings
import urllib.request
import io
import zipfile
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
from sklearn.neighbors import KernelDensity
from scipy.io import mmread


# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

# Optional function to normalize the curvatures to the interval [0, 1]
def normalize_curvatures(curv):
    if curv.max() != curv.min():
        k = 0.001 + (curv - curv.min())/(curv.max() - curv.min())
    else:
        k = curv
    return k

# Compute the free energy
def free_energy():
	n = A.shape[0]
	free_energy = 0
	for i in range(n):
		neighbors = A[i, :]
		indices = neighbors.nonzero()[0]
		labels = node_labels[indices]
		uim = np.count_nonzero(labels==node_labels[i])
		free_energy += uim
	return free_energy

# Defines the pseudo-likelihood function
def pseudo_likelihood(beta):
	n = A.shape[0]
	# Computes the free energy
	free = free_energy()
	# Computes the number of labels (states of the Potts model)
	c = len(np.unique(node_labels))
	# Computes the expected energy
	expected = 0
	for i in range(n):
		neighbors = A[i, :]
		indices = neighbors.nonzero()[0]
		labels = node_labels[indices]
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

# Compute the first and second order Fisher local information
def FisherInformation(A, beta):
	n = A.shape[0]
	# Computes the number of labels (states of the Potts model)
	c = len(np.unique(node_labels))
	PHIs = np.zeros(n)
	PSIs = np.zeros(n)
	for i in range(n):
		neighbors = A[i, :]
		indices = neighbors.nonzero()[0]
		labels = node_labels[indices]
		uim = np.count_nonzero(labels==node_labels[i])
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

# Computes the evaluation metrics
def compute_metrics(G, communities):    
    modula = nx.community.modularity(G, communities)
    coverage, performance = nx.community.partition_quality(G, communities)
    return (modula, coverage, performance)
    

##############################################
############# Beginning of the script
##############################################
# Read the network
#G = nx.karate_club_graph()

#url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"
#name = "football"

#url = "https://websites.umich.edu/~mejn/netdata/dolphins.zip"
#name = "dolphins"

#url = "https://websites.umich.edu/~mejn/netdata/lesmis.zip"
#name = "lesmis"

url = "https://websites.umich.edu/~mejn/netdata/polbooks.zip"
name = "polbooks"

#url = "http://vlado.fmf.uni-lj.si/pub/networks/data/sport/football.net"
#name = "soccer"

#url = "http://vlado.fmf.uni-lj.si/pub/networks/data/mix/USAir97.net"
#name = "USAir97"

#url = "https://nrvis.com/download/data/bio/bio-celegans.zip"
#name = "bio-celegans"

#url = "https://nrvis.com/download/data/bio/bio-diseasome.zip"
#name = "bio-diseasome"

#url = "https://nrvis.com/download/data/eco/eco-everglades.zip"
#name = "eco-everglades"

if 'name' in globals():
    if name == 'football' or name == 'dolphins' or name == 'lesmis' or name == 'polbooks':
        sock = urllib.request.urlopen(url)      # open URL
        s = io.BytesIO(sock.read())             # read into BytesIO "file"
        sock.close()
        zf = zipfile.ZipFile(s)                 # zipfile object
        txt = zf.read(name+".txt").decode()  # read info file
        gml = zf.read(name+".gml").decode()  # read gml data
        # throw away bogus first line with # from mejn files
        gml = gml.split("\n")[1:]
        G = nx.parse_gml(gml)  # parse gml data
    elif name == 'netsience':
        sock = urllib.request.urlopen(url)      # open URL
        s = io.BytesIO(sock.read())             # read into BytesIO "file"
        sock.close()
        zf = zipfile.ZipFile(s)
        zf.extract(name+'.net')
        G = nx.read_pajek(name+'.net')
    elif name == 'soccer' or name == 'USAir97':
        urllib.request.urlretrieve(url, name+'.net')
        H = nx.read_pajek(name+'.net')
        G = H.to_undirected()
    elif name == 'bio-celegans' or name == 'bio-diseasome':
        sock = urllib.request.urlopen(url)      # open URL
        s = io.BytesIO(sock.read())             # read into BytesIO "file"
        sock.close()
        zf = zipfile.ZipFile(s)                 # zipfile object
        zf.extract(name+'.mtx')
        a = mmread(name+'.mtx')
        G = nx.Graph(a)
    elif name == 'eco-everglades':
        sock = urllib.request.urlopen(url)      # open URL
        s = io.BytesIO(sock.read())             # read into BytesIO "file"
        sock.close()
        zf = zipfile.ZipFile(s)                 # zipfile object
        zf.extract(name+'.edges')
        a = nx.read_edgelist(name+'.edges', data=False)
        G = nx.Graph(a)
    
##############################################
######### Beginning of the script
#############################################
# Number of nodes
n = len(G.nodes())
# Number of edges
m = len(G.edges())
# print info
print('Number of nodes: ', n)
print('Number of edges:', m)
print()

# Print input network
pos = nx.spring_layout(G)
#pos = nx.kamada_kawai_layout(G)
plt.figure(1)
nx.draw_networkx(G, pos, with_labels=False, node_size=50, width=0.25, alpha=0.5)
plt.show()

# Community detection
communities = nx.community.greedy_modularity_communities(G)
#communities = nx.community.louvain_communities(G)

# List of colors
number_of_colors = len(list(communities))
colors = ['blue', 'red', 'green', 'black', 'orange', 'magenta', 'darkkhaki', 'brown', 'purple', 'cyan', 'salmon', 'cornflowerblue', 'tomato', 'silver', 'lime', 'seagreen', 'lightcyan', 'teal', 'violet', 'darkviolet', 'pink', 'skyblue', 'chocolate', 'bisque', 'tan', 'lightgreen'][:number_of_colors]
node_colors = []
for node in G:
    current_community_index = 0
    for community in communities:
        if node in community:
            node_colors.append(colors[current_community_index])
            break
        current_community_index += 1

print('Number of communities: ', number_of_colors)

# Compute the metrics
modula, coverage, performance = compute_metrics(G, communities)
print('\nModularity (original network): ', modula)
print('Coverage (original network): ', coverage)
print('Performance (original network): ', performance)
print()

# Plot the communities
plt.figure(2)
nx.draw_networkx(G, pos, with_labels=False, node_size=50, node_color=node_colors, width=0.25, alpha=0.5)

conductancies = []
for i in range(len(communities)):
    for j in range(i+1, len(communities)):
        cond = nx.cuts.conductance(G, communities[i], communities[j])
        conductancies.append(cond)
print('Average conductancea between communities in the network: %f' %(sum(conductancies)/len(conductancies)))
print('Maximum conductancea between communities in the network: %f' %(max(conductancies)))
print()

# Adjacency matrix and node labels (colors codes won't work, it must be integer!)
A = nx.to_numpy_array(G)
node_labels = []
for i in range(n):
    for j in range(len(colors)):
        if node_colors[i] == colors[j]:
            break
    node_labels.append(j)
# Convert to numpy array
node_labels = np.array(node_labels)

# Estimates the maximum pseudo-likelihood estimator of the inverse temperature
sol = optimize.root_scalar(pseudo_likelihood, method='secant', x0=0, x1=1)
# Maximum pseudo-likelihood estimator
critical_beta = np.log(1+np.sqrt(number_of_colors))

print('Critical beta:', critical_beta)
print('MPL beta estimator: ', sol.root)
print()

beta = min(critical_beta, sol.root)

# Compute the first and second order local Fisher information
PHI, PSI = FisherInformation(A, beta)
# Approximate the local curvatures
curvaturas = -PSI/(PHI+0.001)
# Normalize curvatures
K = normalize_curvatures(curvaturas)
# Threshold
limiar = np.quantile(K, 0.8)

# Define the low and high information nodes 
node_colors_th = node_colors.copy()
for i in range(n):
    if K[i] < limiar:
        K[i] = 0
        # Low information nodes
        node_colors_th[i] = 'cornflowerblue'
    else:
        # High information nodes
        node_colors_th[i] = 'tomato'

# Plot the high information points
plt.figure(3)
nx.draw_networkx(G, pos, with_labels=False, node_size=50, node_color=node_colors_th, width=0.25, alpha=0.75)
plt.show()

# Decompose in L and H (K)
L_nodes = np.where(K==0)[0]
L_colors = np.array(node_colors)[L_nodes]
H_nodes = np.where(K>0)[0]
H_colors = np.array(node_colors)[H_nodes]

try:
    cond = nx.cuts.conductance(G, L_nodes, H_nodes)
    print('Conductance between L and H nodes: ', cond)
except ZeroDivisionError:
    print('Impossible to compute the conductancy between L e H nodes')
print()

# Remove nodes to generate L-subgraph and H-subgraph
L = G.copy()
H = G.copy()
L.remove_nodes_from(np.array(list(L.nodes()))[H_nodes])
H.remove_nodes_from(np.array(list(H.nodes()))[L_nodes])

# Plot L and H nodes
plt.figure(5)
nx.draw_networkx(L, pos, with_labels=False, node_size=50, node_color=L_colors, width=0.25, alpha=0.5)
plt.figure(6)
nx.draw_networkx(H, pos, with_labels=False, node_size=50, node_color=H_colors, width=0.25, alpha=0.5)
plt.show()

# Compute the metrics in L
L_labels = node_labels[L_nodes]
labels = np.unique(L_labels)
new_communities  = []
for r in labels:
    indices = np.where(L_labels==r)[0]
    new_communities.append(np.array(list(L.nodes()))[indices])

modula, coverage, performance = compute_metrics(L, new_communities)
print('Modularity (L-subgraph): ', modula)
print('Coverage (L-subgraph): ', coverage)
print('Performance (L-subgraph): ', performance)
print()   

# Calcula modularidade de H
H_labels = node_labels[H_nodes]
labels = np.unique(H_labels)
new_communities = []
for r in labels:
    indices = np.where(H_labels==r)[0]
    new_communities.append(np.array(list(H.nodes()))[indices])

try:
    modula = nx.community.modularity(H, new_communities)
    coverage, performance = nx.community.partition_quality(H, new_communities)
    print('Modularity (H-subgraph): ', modula)
    print('Coverage (H-subgraph): ', coverage)
    print('Performance (H-subgraph): ', performance)
    print()
    H_spec = True
except ZeroDivisionError:
    print('Impossible to compute the metrics in the H-subgraph')
    H_spec = False

# Plot spectrum
L_spectrum = np.real(nx.laplacian_spectrum(L))
H_spectrum = np.real(nx.laplacian_spectrum(H))

plt.figure(7)
kdeplot(L_spectrum, bw_adjust=0.25, color='blue')
plt.title('L spectrum')

if H_spec:
    plt.figure(8)
    kdeplot(H_spectrum, bw_adjust=0.25, color='red')
    plt.title('H spectrum')
    
plt.show()
