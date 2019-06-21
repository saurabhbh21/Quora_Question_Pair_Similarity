FROM tensorflow/tensorflow:latest-gpu-py3

# Install uWSGI
RUN pip3 install uwsgi

#prerequisites of nginx
RUN apt install curl -y gnupg2 ca-certificates lsb-release

#To set up the apt repository for stable nginx packages
RUN echo "deb http://nginx.org/packages/ubuntu `lsb_release -cs` nginx" | tee /etc/apt/sources.list.d/nginx.list

#import an official nginx signing key so apt could verify the packages authenticity
RUN curl -fsSL https://nginx.org/keys/nginx_signing.key | apt-key add -

#Verify that you now have the proper key
RUN apt-key fingerprint ABF5BD827BD9BF62

#install nginx
RUN apt -y update
RUN apt install -y nginx


#create user nginx
#RUN useradd --no-create-home nginx

# Remove default configuration from Nginx
RUN rm /etc/nginx/conf.d/default.conf

# Copy the base uWSGI ini file to enable default dynamic uwsgi process number
COPY ./config/uwsgi.ini /etc/uwsgi/

# Install Supervisord
RUN apt-get update && apt-get install -y supervisor && rm -rf /var/lib/apt/lists/*

# Custom Supervisord config
COPY ./config/supervisord.conf /etc/supervisor/conf.d/supervisord.conf


# Which uWSGI .ini file should be used, to make it customizable
ENV UWSGI_INI /config/uwsgi.ini

# By default, run 2 processes
ENV UWSGI_CHEAPER 2

# By default, when on demand, run up to 16 processes
ENV UWSGI_PROCESSES 16

# By default, allow unlimited file sizes, modify it to limit the file sizes
# To have a maximum of 1 MB (Nginx's default) change the line to:
# ENV NGINX_MAX_UPLOAD 1m
ENV NGINX_MAX_UPLOAD 0

# By default, Nginx will run a single worker process, setting it to auto
# will create a worker for each CPU core
ENV NGINX_WORKER_PROCESSES 1

# By default, Nginx listens on port 5000
ENV LISTEN_PORT 5000


# Add the project and set working directory
COPY . /
WORKDIR /



# scripts/start.sh will check for a /scripts/download.sh script and run it before starting the app 
# give executable permission to scripts/start.sh
RUN chmod +x /scripts/start.sh

# Executable permission to entrypoint that will generate Nginx additional configs
#COPY ./scripts/uwsgi-nginx-entrypoint.sh /scripts/uwsgi-nginx-entrypoint.sh
RUN chmod +x /scripts/uwsgi-nginx-entrypoint.sh

#install the requirements
RUN pip3 install -r requirements.txt

ENTRYPOINT ["/scripts/uwsgi-nginx-entrypoint.sh"]