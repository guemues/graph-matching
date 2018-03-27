This repository created for a semester project at EPFL.

Feedback 1:

Hello Orçun,

My Github username is: trouleau

And by the way, you have access to room BC 255 if you want a place to work. You can go there whenever you want, your Camipro card should open the door. Otherwise I can show you around whenever you want.


Here are a couple more feedback following our meeting (some of it we already discussed):

- Shuffle the nodes before computing embeddings

- Try higher dimensions for the embeddings (maybe that is the reason the locally linear embedding was performing bad)

- On your knn plot, also check if the distance of the nodes are important (the intuition is that the probability of correct match also depends on the distance)

- For the noise procedure, you should actually do as follows:

- Take a master graph

- Generate a noisy graph by sampling edges from the master with probability (1-q)

- In the graph alignment tasks, you should not compare a “noisy graph” to a “master graph”, but two different versions of “noisy graphs” (this is a valid comment for the knn plot you showed me)


I can give you more clarifications if something is not clear.

Best,
Wiliam


Feedback 2:

1. Forget about GNP graphs, try graphs with powerlaw degree distribution.

2. For high dimension, make sure columns don’t get swapped

3. KNN experiment   

   a.  also, Given threshold, compute the accuracy as a function of noise (i.e. is the matching node in a ball of radius the threshold)

   b. Same thing but with a given a noise value, compute the accuracy as a function of threshold

   c. Keep track of the average number of nodes in the ball as a function of the threshold

4. Compare the accuracy of the matching with some node features (e.g. node degree, 

5. Read the embedding-based graph alignment paper and understand the algorithm (no need to implement it for now)

6. Write a report on all the results produced up to now with short explanation and plots.


Feedback 3:

I prepared these notes from last week: 

* Clear notation for accuracy and for he other notations which are not well defined (true positive, false positives etc) 

	1- in the report 
	2- presentation
	
	with mathematical notations
	with visualizations
	
* Show accuracy in respect to node degree.

* A new notation for embedding:
	
	1- Create your own embedding via neighbors node degrees by creating histogram bins.
	2- Eliminate low degree nodes calculate accuracy for high degree nodes. 
	3- And maybe combine the histogram features and the embedding features from previous testes to increase the accuracy

These are feedback from last week.


Feedback 4:

* clean plots (make sure all params are explicits on the graphs)

* average over multiple noisy graphs (mean variance)

* node degree + degree of neig of neig

* read paper


