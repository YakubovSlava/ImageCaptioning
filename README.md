This repository has everything for installation your app on WS.
First of all u need to clone all files from this repo on your machine. Secondly, download and install python 3 on your UBUNTU.
Third step is to create python environment, which allows to simplify usage of pip3.
After you have done all of these steps, just use command "`pip install -r requirements.txt`" this command will download all important libraries. Also use command
`pip install gunicorn`. Gunicorn will help to run your flask app.
You are ready to start.
`gunicorn -w n -b 0.0.0.0:p main:app --reload` starts application.
p is port you want to use.
w is number of workers which listen your port the same time (how many people can use your app).
**BE CAREFUL** if u use lots of workers, it can use all of your RAM.
**it is ready!.**

