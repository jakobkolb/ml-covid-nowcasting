notebook:
	poetry run jupyter-notebook

remote-notebook-server:
	poetry run jupyter notebook --no-browser --port=8090

remote-notebook-connect:
	ssh -L 8090:localhost:8090 kolb@ornt.biologie.hu-berlin.de