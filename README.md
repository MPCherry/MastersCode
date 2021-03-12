# IntersectionManagement

Matthew P. Cherry, 2021.

## Prerequisites

Java 11 and Maven must be installed.

https://www.oracle.com/za/java/technologies/javase-jdk11-downloads.html
https://maven.apache.org/download.cgi

## Compile

```mvn clean package```

## Evolve a controller

```java -jar Evolution/target/Evolution-1.0-SNAPSHOT-jar-with-dependencies.jar neat experiments/test.json```

Instead of ```neat```, two other values are allowed, ```cne``` or ```hyperneat```.

Any experiment can be specified from the ```Evolution/src/main/resources/experiments``` folder.

Different experiments from the 

## Run the simulator

```java -Djava.library.path=Trial/target/natives -jar Trial/target/Trial-1.0-SNAPSHOT-jar-with-dependencies.jar sample_parameters.json```

Edit ```sample_parameters.json``` to try out different tracks, controllers and traffic configurations.