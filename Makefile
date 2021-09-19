
codecheck:
	@isort --check .
	@black --check -S .
	@bandit --recursive .
