\name{find.log.alpha}
\alias{find.log.alpha}
\title{Computes the water-filling level, alpha, with which
samples, and their weights, are deterministically accepted.}
\usage{
find.log.alpha(logWeights, logWeightsSum = NULL, N)
}
\arguments{
  \item{logWeights}{vector of log weights}

  \item{logWeightsSum}{total log weight, i.e. log sum of
  weights.  The value NULL implies that the sum is to be
  computed by \code{find.log.alpha}.}

  \item{N}{the number of samples.  Must be <= the length of
  logWeights.}
}
\value{
numeric value of alpha
}
\description{
Computes the water-filling level, alpha, with which
samples, and their weights, are deterministically accepted.
}
\details{
For details concerning the algorithm see the paper by
Nicholas Polson, Brandon Willard (2014).
}
\examples{
x = seq(0, 10, 1)
  log.weights = dt(x, df=1, log=T)
  log.weights.sum = log(sum(dt(x, df=1, log=F)))

  log.alpha = find.log.alpha(log.weights, log.weights.sum, 8)
  print(log.alpha)
}
\author{
Brandon Willard \email{brandonwillard@gmail.com}
}
\references{
Nicholas G. Polson, Brandon Willard (2014), "Recursive
Bayesian Computation".
}
\seealso{
\code{\link{water.filling.resample}}
}
\keyword{water-filling}

