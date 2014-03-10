# Borrowed from http://cran.r-project.org/web/packages/OpenStreetMap/index.html
# 
# Author: brandonwillard
###############################################################################

.isMac <- function(){
	length(grep("^darwin",R.version$os))>0
}

.tryJava <- function(){
	ty <- try(new(J("com.statslibextensions.statistics.ExtSamplingUtils")))
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


.java.check.ex.print.stack <- function() {
  if (!is.null(e<-.jgetEx())) {
    write("Java exception was raised", stderr())
    baos <- new(J("java.io.StringWriter"))
    ps <- .jnew("java.io.PrintWriter", 
        .jcast(baos, new.class="java/io/Writer"))
    e$printStackTrace(ps)
    write(baos$toString(), stderr())
    return(TRUE)
  } else {
    return(FALSE)
  }
}

#
# override to get full stack traces
#
.jcheck <- function(silent=FALSE) {
  if(.java.check.ex.print.stack())
    write("(overridden .jcheck)", stderr())
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
  
  
  unlockBinding(".jcheck", as.environment("package:rJava"))
  assign(".jcheck", .jcheck, as.environment("package:rJava"))
  lockBinding(".jcheck", as.environment("package:rJava"))
  unlockBinding(".jcheck", getNamespace("rJava"))
  assign(".jcheck", .jcheck, getNamespace("rJava"))
  lockBinding(".jcheck", getNamespace("rJava"))
  
}

