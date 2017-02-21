test : download-test-data
	python setup.py test

test-all : download-test-data
	python setup.py test --addopts '--runslow'

coverage-run : download-test-data
	coverage run setup.py test --addopts '--runslow'

coverage : coverage-run
	coverage report

py3-lint : 
	pylint --py3k ./plutokore

yapf : 
	yapf -ip --style pep8 --recursive ./plutokore
	yapf -ip --style pep8 --recursive ./tests

tests/data/pluto/pluto.ini :
	wget -qO- "https://drive.google.com/uc?export=download&id=0B-nHmohemH-ZcVFCdThlRF9wTG8" | tar xvz -C ./tests/data

download-test-data : tests/data/pluto/pluto.ini

.PHONY: test
