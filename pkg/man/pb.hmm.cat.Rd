\name{pb.hmm.cat}
\alias{pb.hmm.cat}
\title{Particle filter for a Hidden Markov Model with categorical emissions.}
\usage{
pb.hmm.cat(y, M, X, hmmClassProbs, hmmTransProbs, emissionProbs,
  numParticles = 1000, seed = NULL)
}
\arguments{
  \item{y}{dependent variable vector}

  \item{X}{explanatory variable matrix}

  \item{M}{number of}

  \item{hmmClassProbs}{vector of initial HMM state
  probabilities}

  \item{hmmTransProbs}{matrix of HMM state stransitions}

  \item{emissionProbs}{vectors of probabilities for
  observable categories in each HMM state.  Rows span
  components, columns span observation categories}

  \item{numParticles}{number of particles}

  \item{seed}{seed for the random number generator.}
}
\value{
A list containing a matrix of estimated HMM states and
corresponding a vector of weights, for each particle and
time
}
\description{
Particle filter for a Hidden Markov Model with categorical
emissions.
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
\keyword{hidden}
\keyword{markov}
\keyword{model}
\keyword{water-filling}

