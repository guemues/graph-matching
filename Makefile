deploy:
	scp -r /Users/guemues/Desktop/epfl/graph-matching guemues@iccluster051.iccluster.epfl.ch:/home/guemues/

pull:
	scp -r guemues@iccluster051.iccluster.epfl.ch:/home/guemues/graph-matching/results /Users/guemues/Desktop/epfl/graph-matching