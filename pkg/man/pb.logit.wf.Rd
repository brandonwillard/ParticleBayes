\name{pb.logit.wf}
\alias{pb.logit.wf}
\title{Particle filter implementing binomial logistic regression via an EV mixture
of normals approximation and water-filling.}
\usage{
pb.logit.wf(y, X, m0 = rep(0, ncol(X)), C0 = diag(ncol(X)),
  G = diag(nrow(C0)), W = diag(nrow(C0)), numParticles = 1000,
  seed = NULL, parallel = TRUE)
}
\arguments{
  \item{y}{dependent variable vector}

  \item{X}{explanatory variable matrix}

  \item{m0}{initial prior mean vector for the state
  variable}

  \item{C0}{initial prior covariance matrix for the state
  variable}

  \item{G}{constant state evolution matrix}

  \item{W}{constant state evolution covariance matrix}

  \item{numParticles}{number of particles}

  \item{seed}{seed for the random number generator.}
}
\value{
A list containing each resampled object in the support and
its associated weight.
}
\description{
Particle filter implementing binomial logistic regression
via an EV mixture of normals approximation and
water-filling.
}
\details{
For details concerning the algorithm see the paper by
Nicholas Polson, Brandon Willard (2014).
}
\author{
Brandon Willard \email{brandonwillard@gmail.com}
}
\references{
Nicholas G. Polson, Brandon Willard (2014), "Recursive
Bayesian Computation". Sylvia Fruehwirth-Schnatter and
Rudolf Fruehwirth (2010), "Data augmentation and MCMC for
binary and multinomial logit models." In \emph{Statistical
Modelling and Regression Structures - Festschrift in Honour
of Ludwig Fahrmeir}, T. Kneib and G. Tutz, Eds.
Physica-Verlag, Heidelberg, pp. 111-132.
}
\keyword{logistic}
\keyword{regression}
\keyword{water-filling}

