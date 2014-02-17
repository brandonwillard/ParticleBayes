\name{pb.logit.wf}
\alias{pb.logit.wf}
\title{Particle filter implementing binomial logistic regression via a mixture
of normals and using water-filling.}
\usage{
pb.logit.wf(y, X, m0 = rep(0, ncol(X)), C0 = diag(ncol(X)),
  G = diag(nrow(C0)), W = diag(nrow(C0)), numParticles = 1000,
  waterFilling = T, seed = NULL)
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

  \item{waterFilling}{boolean value determining whether or
  not to perform water-filling}

  \item{seed}{seed for the random number generator.}
}
\value{
A list containing each resampled object in the support and
its associated weight.
}
\description{
Particle filter implementing binomial logistic regression
via a mixture of normals and using water-filling.
}
\details{
For details concerning the algorithm see the paper by
Nicholas Polson, Brandon Willard (2014).
}
\author{
Brandon Willard \email{brandonwillard@gmail.com}
}
\references{
Sylvia Fruehwirth-Schnatter and Rudolf Fruehwirth (2010),
"Data augmentation and MCMC for binary and multinomial
logit models." In \emph{Statistical Modelling and
Regression Structures - Festschrift in Honour of Ludwig
Fahrmeir}, T. Kneib and G. Tutz, Eds. Physica-Verlag,
Heidelberg, pp. 111-132.

Nicholas G. Polson, Brandon Willard (2014), "Recursive
Bayesian Computation".
}
\keyword{logistic}
\keyword{regression}
\keyword{water-filling}

