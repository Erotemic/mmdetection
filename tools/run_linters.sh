flake8 .

isort -rc --check-only --diff mmdet/ tools/ tests/

yapf -r -d --style .style.yapf mmdet/ tools/ tests/


##### 

yapf -r -i --style .style.yapf mmdet/ tools/ tests/

isort --apply -rc mmdet/ tools/ tests/
