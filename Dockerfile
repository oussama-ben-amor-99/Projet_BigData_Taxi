FROM apache/hadoop-runner:jdk11-u2204

ARG HADOOP_URL=https://dlcdn.apache.org/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz
WORKDIR /opt
RUN sudo apt update && sudo apt install -y curl
RUN sudo rm -rf /opt/hadoop && sudo curl -LSs -o hadoop.tar.gz $HADOOP_URL && sudo tar zxf hadoop.tar.gz && sudo rm hadoop.tar.gz && sudo mv hadoop* hadoop && sudo rm -rf /opt/hadoop/share/doc
WORKDIR /opt/hadoop
# On crée le fichier directement ici pour éviter les erreurs de copie
RUN echo "hadoop.root.logger=INFO,console" > /opt/hadoop/etc/hadoop/log4j.propertiesRUN sudo chown -R hadoop:users /opt/hadoop/etc/hadoop/*
ENV HADOOP_CONF_DIR /opt/hadoop/etc/hadoop
RUN sudo apt install -y nano 