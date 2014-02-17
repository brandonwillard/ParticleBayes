ParticleBayes
=====
ParticleBayes is an R package that contains a collection of 
[particle filters](http://en.wikipedia.org/wiki/Particle_filter) 
for a subset of hierarchical bayesian models, with a focus on sequentially 
learning model parameters alongside state variables.  The R code calls an
open-source Java library, [ParticleLearningModels][plm], to perform calculations, 
so when/if a practitioner wants to implement a model in production, 
the underlying, streaming-capable, API is available.
  
Most models are formulated as [dynamic linear models][dlm], or mixtures thereof,
although some support for hidden markov models exists.
The models also implement  

* [Water-filling resampling][waterfilling]
* Parameter learning, via [Particle Learning][pl] where applicable

Currently, the implemented filters cover two response types:  

* Multinomial/Categorical with a logistic link function  
 * 10-component [normal mixture approximation][fs1]  
 * [TBD][fs2]  
* Multivariate Gaussian  

and, naturally, their univariate equivalents.

Installation from R
==
ParticleBayes depends on the rJava R package, so make sure it's installed and
operational.  This also means you'll need to have a functional JRE, so if you
don't you can get one from 
[here](http://www.oracle.com/technetwork/java/javase/downloads/java-archive-downloads-javase6-419409.html#jre-6u45-oth-JPR), 
install, and then run ```R CMD javareconf``` (with sudo if you're using linux)
at the command line.  If you're using a Mac, you might need to run something
like ```R CMD javareconf JAVA_CPPFLAGS=-I/System/Library/Frameworks/JavaVM.framework/Headers```
instead.  Pay attention for any errors in the output, and look around the internet 
for more information.
 
While ParticleBayes is still in it's beta stages, installation in R requires 
the [devtools][dt] package and something like the following:
```
library(devtools)
install_url("https://bitbucket.org/brandonwillard/particlebayes/get/HEAD.zip", subdir="pkg")
```
then, as usual, add ```library(ParticleBayes)``` to your R source to use.  
NOTE: This will pick up the most recently pushed version of the code (i.e. the
jar file in ```pkg/inst/java```).  Inclusion of this file in the repo is temporary,
that is, until this package gets submitted to CRAN.

Development Setup and Installation
==
ParticleBayes uses [Maven](http://maven.apache.org/), so you'll need to download that.  
Once you have it,
```mvn clean package -Dmaven.test.skip=true```
should create the necessary jars in ```pkg/inst/java```, so that the code is callable from
R.   
The java code is called, within R, through [rJava][rj]

Similar to the R installation, and from the project's root directory, 
one possible development setup script could include
```r
library(devtools)

dev_mode(on=T)

system("mvn clean package -Dmaven.test.skip=true")

install_local("./pkg")

options(java.parameters=
        c("-Xmx2g", 
          "-Xdebug", 
          "-Xrunjdwp:server=y,transport=dt_socket,address=4001,suspend=n"))


library(ParticleBayes)

...

dev_mode(on=F)
```
The script will build the Java code, install the package in an isolated environment 
(see [devtools][dt]), and set up the JVM for debugging. 

[rj]:http://www.rforge.net/rJava/
[dt]:https://github.com/hadley/devtools
[plm]:https://bitbucket.org/brandonwillard/particlelearningmodels
[dlm]:http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.34.9040
[fs1]:http://www.sciencedirect.com/science/article/pii/S0167947306003720
[fs2]:http://dl.acm.org/citation.cfm?id=2414419
[pl]:http://projecteuclid.org/euclid.ss/1280841735
[waterfilling]:tbd
