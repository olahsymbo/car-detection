cd src &
virtualenv -p /usr/local/opt/python@3.7/bin/python3.7 .virtualenv &
pip install -r src/requirements.txt &
source .virtualenv/bin/activate &
gunicorn --bind 0.0.0.0:8080 wsgi:app --workers 8 --threads 4