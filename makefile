all: main

main: main.py
	python3 main.py

run: 
	python3 main.py

tom: 
	python3 main.py tom

gabbo: 
	python3 main.py gabbo

clean: 
	rm -f __pycache__