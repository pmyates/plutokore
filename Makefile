test :
	python setup.py test

test-all :
	python setup.py test --addopts '--runslow'

coverage-run :
	coverage run setup.py test-all

coverage : coverage-run
	coverage report

py3-lint : 
	pylint --py3k ./plutokore

yapf : 
	yapf -ip --style pep8 --recursive ./plutokore
	yapf -ip --style pep8 --recursive ./tests

.PHONY: test
