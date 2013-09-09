# Borrowed from http://cran.r-project.org/web/packages/OpenStreetMap/index.html
# 
# Author: brandonwillard (orginally ianfellows)
###############################################################################

.isMac <- function(){
	length(grep("^darwin",R.version$os))>0
}

.tryJava <- function(){
	ty <- try(new(J("plm.utils.SamplingUtils")))
	if(inherits(ty,"try-error")){
		stop(
"Java classes could not be loaded. Most likely because Java is not set up with your R installation.
Here are some trouble shooting tips:

1. Install Java
2. Run 
\tR CMD javareconf
in the terminal. If you are using Mac OS X >= 10.7 you may want to try
\tR CMD javareconf JAVA_CPPFLAGS=-I/System/Library/Frameworks/JavaVM.framework/Headers
instead.
"
		)
	}
}


.onLoad <- function(libname, pkgname) {
	.jni <- try(get(".jniInitialized"),silent=TRUE)
	if(inherits(.jni,"try-error"))
		.jni <- FALSE
	if(.isMac() && !.jni)
		Sys.setenv(NOAWT=1)

	ty <- try(.jpackage(pkgname, lib.loc=libname) )
	if(inherits(ty,"try-error")){
		stop(
				"Java classes could not be loaded. Most likely because Java is not set up with your R installation.
Here are some trouble shooting tips:
1. Install Java					
2. Run 
\tR CMD javareconf
in the terminal. If you are using Mac OS X >= 10.7 you may want to try
\tR CMD javareconf JAVA_CPPFLAGS=-I/System/Library/Frameworks/JavaVM.framework/Headers
instead."
		)
	}	
}

