\name{pb.hmm.ar1}
\alias{pb.hmm.ar1}
\title{Particle filter for a gaussian AR(1) dynamic linear model modulated by an HMM.
States have normal priors, psi is a joint normal that consists of AR(1) and state offset parameters
and sigma2 has an inverse-gamma prior. Psi and sigma2 are state dependent.}
\usage{
pb.hmm.ar1(y, FF, m0, C0 = diag(length(m0)), mPsi0 = rep(0, 2 * ncol(C0)),
  CPsi0 = diag(2 * ncol(C0)), sigma2Scale = 1, sigma2Shape = 2,
  initialProbs, transProbs, numSubSamples = 3, numParticles = 1000,
  seed = NULL)
}
\arguments{
  \item{y}{dependent variable vector}

  \item{FF}{observation matrix}

  \item{m0}{state prior mean vector}

  \item{C0}{state prior covariance matrix}

  \item{mPsi0}{vector of state transition constant and
  AR(1) prior means}

  \item{CPsi0}{state transition constant and AR(1) prior
  covariance matrix}

  \item{sigma2Scale}{scale parameter for sigma2
  inverse-gamma prior}

  \item{sigma2Shape}{shape parameter for sigma2
  inverse-gamma prior}

  \item{initialClassProbs}{initial class probabilities}

  \item{transProbs}{transition probabilities}

  \item{numSubSamples}{sub samples for state variable}

  \item{numParticles}{number of particles}

  \item{seed}{seed for the random number generator.}
}
\value{
returns a list containing matrices spanning time, particle
and values, where values are logWeights, stateMeans,
stateCovs, psiMeans, psiCovs, sigma2Shapes, and
sigma2Scales
}
\description{
Particle filter for a gaussian AR(1) dynamic linear model
modulated by an HMM. States have normal priors, psi is a
joint normal that consists of AR(1) and state offset
parameters and sigma2 has an inverse-gamma prior. Psi and
sigma2 are state dependent.
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
Bayesian Computation".
}
\keyword{autoregressive}
\keyword{dlm}
\keyword{dynamic}
\keyword{hidden}
\keyword{linear}
\keyword{markov}
\keyword{model}
\keyword{water-filling}

